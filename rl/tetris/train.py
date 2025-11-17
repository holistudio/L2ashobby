import numpy as np
import torch

from pufferlib.ocean.tetris import tetris
from agent import PPOAgent

from utils.mpi_pytorch import setup_pytorch_for_mpi
from utils.mpi_tools import proc_id, num_procs

def train(n_episodes=100, buffer_size=4000, seed=0):
     # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = tetris.Tetris()

    agent = PPOAgent(env.single_observation_space, env.single_action_space,
                     local_steps_per_epoch=buffer_size)
    buf = agent.buffer
    steps = 0
    

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_ret, ep_len =  0, 0
        
        done = False
        total_reward = 0

        while not done:
            # action = env.action_space.sample()
            obs = torch.as_tensor(obs, dtype=torch.float32)
            a, v, logp = agent.step(obs)
            action = a

            next_obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            # frame = env.render() # comment out if you want headless training

            ep_ret += reward
            ep_len += 1
            buf.store(obs, a, reward, v, logp)

            obs = next_obs

            total_reward += reward
            done = terminated or truncated

            if done:
                print(f"Episode {ep} finished! Total reward: {total_reward}, Total Length: {ep_len}")
                if truncated:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    obs = torch.as_tensor(obs, dtype=torch.float32)
                    _, v, _ = agent.step(obs)
                else:
                    v = 0

                buf.finish_path(v)

                (obs, _), ep_ret, ep_len = env.reset(), 0, 0

        if steps >= buffer_size:
            # agent update
            agent.update()
    
    env.close()
    pass

if __name__ == '__main__':
    train()