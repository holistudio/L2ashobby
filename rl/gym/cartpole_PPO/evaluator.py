import gymnasium as gym
import numpy as np
import eval_env

import torch
from agent import PPOAgent

class CartPoleEvaluator:
    def __init__(self):
        self.headless_env: gym.Env[np.ndarray, int] = gym.make(
            "CartPoleCustom-v0",
            max_episode_steps=5000,
        )
        self.rendered_env: gym.Env[np.ndarray, int] = gym.make(
            "CartPoleCustom-v0",
            max_episode_steps=5000,
            render_mode="human",
        )
        # TODO: init your agent here
        self.agent = PPOAgent(self.headless_env.observation_space, self.headless_env.action_space)
        self.agent.load()

    def select_action(self, state: np.ndarray) -> int:
        # TODO: implement me!
        a, v, logp = self.agent.step(torch.as_tensor(state, dtype=torch.float32))
        return a

    def num_parameters(self) -> int:
        # TODO: implement me!
        print(self.agent.mlp_ac)
        return sum(p.numel() for p in self.agent.mlp_ac.parameters())

    def evaluate(self, render_mode: str = 'human', n_episodes: int = 10):
        n_parameters = self.num_parameters()
        print(f"agent parameters: {n_parameters}")
        rewards: list[float] = []
        for i in range(n_episodes):
            if render_mode == 'human':
                env = self.rendered_env
            elif render_mode == 'headless':
                env = self.headless_env
            else:
                raise ValueError('Invalid render_mode')
            state, _ = env.reset()
            ended = False
            episode_reward = 0.0
            while not ended:
                action = self.select_action(state)
                state, reward, done, terminated, _ = env.step(action)
                ended = done or terminated
                episode_reward += float(reward)
            rewards.append(episode_reward)
        r_mean = np.mean(rewards)
        r_std = np.std(rewards)
        print(f"Mean reward: {r_mean} +/- {r_std}")

        r_mean_pp = r_mean / n_parameters
        r_std_pp = r_std / n_parameters
        print(f"Mean reward per parameter: {r_mean_pp} +/- {r_std_pp}")
        env.close()


if __name__ == "__main__":
    evaluator = CartPoleEvaluator()
    evaluator.evaluate('headless')
