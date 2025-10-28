import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import tqdm

from agent import PPOAgent

from utils.mpi_tools import num_procs

def train(n_episodes=50, steps_per_ep=4000, max_ep_len=1000):
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    agent = PPOAgent(env.observation_space, env.action_space)

    # Set up experience buffer
    local_steps_per_ep = int(steps_per_ep / num_procs())
    buf = agent.buffer

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
            if type(o) == tuple:
                o = o[0]

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            ep_ended = t==local_steps_per_ep-1

            if terminal or ep_ended:
                if ep_ended and not(terminal):
                    print('Warning: trajectory cut off by episode at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or ep_ended:
                    _, v, _ = agent.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    print(f'episode Reward={ep_ret}, episode Length={ep_len}')
                o, ep_ret, ep_len = env.reset(), 0, 0
                if type(o) == tuple:
                    o = o[0]
        
        # PPO agent update
        agent.update()

        # TODO: save agent 

if __name__ == '__main__':
    train()