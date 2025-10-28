import numpy as np
import torch

import gymnasium as gym
from tqdm import tqdm

from agent import PPOAgent

from utils.mpi_pytorch import setup_pytorch_for_mpi
from utils.mpi_tools import proc_id, num_procs

def train(n_episodes=10, steps_per_ep=4000, max_ep_len=1000, seed=0, ac_kwargs=dict()):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1", max_episode_steps=max_ep_len)

    agent = PPOAgent(env.observation_space, env.action_space, **ac_kwargs)

    # Set up experience buffer
    local_steps_per_ep = int(steps_per_ep / num_procs())
    buf = agent.buffer

    # Prepare for interaction with environment
    (o, _), ep_ret, ep_len = env.reset(), 0, 0
    
    # Main loop: collect experience in env and update/log each episode
    for episode in tqdm(range(n_episodes)):
        for t in range(local_steps_per_ep):
            a, v, logp = agent.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ , _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            ep_ended = t==local_steps_per_ep-1

            if terminal or ep_ended:
                # if ep_ended and not(terminal):
                #     print('Warning: trajectory cut off by episode at %d steps.'%ep_len, flush=True)

                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or ep_ended:
                    _, v, _ = agent.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0

                buf.finish_path(v)

                # if terminal:
                #     print(f'episode Reward={ep_ret}, episode Length={ep_len}')

                (o, _), ep_ret, ep_len = env.reset(), 0, 0
        
        # agent update
        agent.update()
    
    env.close()
    
    # TODO: save agent 

if __name__ == '__main__':
    train()