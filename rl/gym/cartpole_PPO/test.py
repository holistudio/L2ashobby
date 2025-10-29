import numpy as np
import torch

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

from agent import PPOAgent

from utils.mpi_pytorch import setup_pytorch_for_mpi
from utils.mpi_tools import proc_id, num_procs

def test(n_episodes=100, steps_per_ep=4000, max_ep_len=1000, seed=0, ac_kwargs=dict()):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Random seed
    # seed += 10000 * proc_id()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    env = gym.make("CartPole-v1", max_episode_steps=max_ep_len)

    agent = PPOAgent(env.observation_space, env.action_space, **ac_kwargs)

    agent.load()


    # Reset environment to start a new episode
    (o, _) = env.reset()

    episode_over = False
    total_reward = 0
    episode_length = 0
    ep_rewards = np.zeros((n_episodes))
    ep_lens = np.zeros((n_episodes))

    # Main loop: collect experience in env and update/log each episode
    for episode in tqdm(range(n_episodes)):
        while not episode_over:
            a, v, logp = agent.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated , _ = env.step(a)
            episode_length += 1
            
            # Update obs (critical!)
            o = next_o

            # reward: +1 for each step the pole stays upright
            # terminated: True if pole falls too far (agent failed)
            # truncated: True if we hit the time limit (500 steps)
            total_reward += r
            episode_over = terminated or truncated

        # print(episode)
        # print(total_reward)
        ep_rewards[episode] = total_reward
        ep_lens[episode] = episode_length
        (o, _), total_reward, episode_length = env.reset(), 0, 0

        episode_over = False
    
    env.close()

    print(ep_rewards)
    print(f"Average rewards={np.mean(ep_rewards)}")

    return agent

if __name__ == '__main__':
    agent = test(max_ep_len=5000)