import numpy as np
import torch
import torch.nn as nn

from pufferlib.ocean.tetris import tetris
from agent import PPOAgent

from utils.mpi_pytorch import setup_pytorch_for_mpi

def print_state(obs):
    print()
    print('Game Grid')
    print(obs[0][:200])
    print()
    print('+ Game States')
    print(obs[0][200:206])
    print()
    print('Tetris Pieces')
    print('Current Piece')
    print(obs[0][206:206+7])
    print('Preview 1')
    print(obs[0][206+7:206+14])
    print('Preview 2')
    print(obs[0][206+14:206+21])
    print('Hold Piece')
    print(obs[0][206+21:206+28])
    print()
    print('Noise')
    print(obs[0][-10:])
    pass

def train(n_episodes=500, buffer_size=4000, seed=0, print_every=50):
     # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = tetris.Tetris()

    agent = PPOAgent(env.single_observation_space, env.single_action_space,
                     local_steps_per_epoch=buffer_size,
                     hidden_sizes=(1024,1024), activation=nn.ReLU, device=device)
    print(agent.mlp_ac.pi.logits_net)
    print()

    buf = agent.buffer
    steps = 0
    


    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_ret, ep_len =  0, 0
        
        done = False
        total_reward = 0

        while not done:
            # action = env.action_space.sample()
            # print_state(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            a, v, logp = agent.step(obs)
            action = a.cpu()

            next_obs, r, terminated, truncated, info = env.step(action)
            steps += 1
            # frame = env.render() # comment out if you want headless training

            reward = r[0].item()
            ep_ret += reward
            ep_len += 1
            buf.store(obs, a, reward, v, logp)

            obs = next_obs

            total_reward += reward
            done = terminated or truncated

            if done:
                
                if truncated:
                    print(f"Episode {ep} truncated!")
                    # if trajectory didn't reach terminal state, bootstrap value target
                    obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
                    _, v, _ = agent.step(obs)
                else:
                    v = 0

                buf.finish_path(v)

                if (ep) % print_every == 0 or ep == (n_episodes-1): 
                    print(f"Episode {ep} | Total reward: {total_reward:.2f}, Game length: {ep_len} timesteps")
                (obs, _), ep_ret, ep_len = env.reset(), 0, 0

        if steps >= buffer_size:
            # agent update
            agent.update()
    
    env.close()
    pass

if __name__ == '__main__':
    print()
    train(n_episodes=1, print_every=50)