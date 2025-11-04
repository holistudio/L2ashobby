import numpy as np
import torch
from tqdm import tqdm

from pettingzoo.classic import connect_four_v3
# from pettingzoo.classic import go_v5

from agent import PPOAgent

from utils.mpi_pytorch import setup_pytorch_for_mpi
from utils.mpi_tools import proc_id, num_procs

def train(n_episodes=100, seed=0, ac_kwargs=dict()):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # env = connect_four_v3.env(render_mode="human")
    env = connect_four_v3.env()
    # env = go_v5.env(board_size = 19, komi = 7.5, render_mode="human")

    local_steps = 100
    agent1 = PPOAgent(env.observation_spaces['player_0']['observation'], env.action_spaces['player_0'], local_steps_per_epoch=local_steps, **ac_kwargs)
    agent2 = PPOAgent(env.observation_spaces['player_1']['observation'], env.action_spaces['player_1'], local_steps_per_epoch=local_steps, **ac_kwargs)

    # Set up experience buffer
    buf1 = agent1.buffer
    buf2 = agent2.buffer

    agent_list = [agent1, agent2]
    buf_list = [buf1, buf2]
    steps = [0, 0] # track steps for each agent's buffer
    idx = 0

    moves = np.zeros(n_episodes)
    
    for episode in tqdm(range(n_episodes)):
        env.reset(seed=42)
        for a in env.agent_iter():
            agent = agent_list[idx]
            buf = buf_list[idx]

            observation, reward, termination, truncation, info = env.last()

            # invalid action masking is optional and environment-dependent
            if "action_mask" in info:
                mask = info["action_mask"]
            elif isinstance(observation, dict) and "action_mask" in observation:
                mask = observation["action_mask"]
            else:
                mask = None

            # blank spaces are 0, +1 represents current player's tokens, -1 represent's opponent player's tokens
            # environment alternates the observation planes for you
            o1 = observation["observation"][:, :, 0]
            o2 = observation["observation"][:, :, 1]
            o = torch.as_tensor(o1-o2, dtype=torch.float32).view(-1) # assumes players pieces do not overlap on the board (true for connect four)
            
            if termination or truncation:
                action = None
                if truncation:
                    _, value, _ = agent.step(o, mask)
                else:
                    value = 0

                buf.finish_path(value)
            else:
                action, value, logp = agent.step(o, mask)
                
                buf.store(o.numpy(), action, reward, value, logp)
                steps[idx] += 1

            env.step(action)
            moves[episode] += 1
            
            idx += 1
            if idx >= len(agent_list):
                idx = 0
            
        
        idx = 0
        # agent update only after the buffer is full
        if steps[0] > local_steps:
            agent1.update()
            steps[0] = 0
        if steps[1] > local_steps:
            agent2.update()
            steps[1] = 0

    env.close()
    print(f"Average number of moves = {moves.mean()}")

    agent1.save(module_filename='Conn4-P1_agent.pt.tar', buffer_filename="Conn4-P1_buffer.npz")
    agent2.save(module_filename='Conn4-P2_agent.pt.tar', buffer_filename="Conn4-P2_buffer.npz")

    return agent1, agent2

if __name__ == '__main__':
    agent1, agent2 = train(n_episodes=10)

    env = connect_four_v3.env(render_mode="human")

    env.reset(seed=42)

    agent_list = [agent1, agent2]
    idx = 0

    
    for a in env.agent_iter():
        agent = agent_list[idx]

        observation, reward, termination, truncation, info = env.last()

        # invalid action masking is optional and environment-dependent
        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
        else:
            mask = None
        
        # blank spaces are 0, +1 represents current player's tokens, -1 represent's opponent player's tokens
        # environment alternates the observation planes for you
        o1 = observation["observation"][:, :, 0]
        o2 = observation["observation"][:, :, 1]
        o = torch.as_tensor(o1-o2, dtype=torch.float32).view(-1) # assumes players pieces do not overlap on the board (true for connect four)

        if termination or truncation:
            action = None
            env.reset()
            break
        else:
            action = agent.act(o, mask)
            env.step(action)
            idx += 1
            if idx >= len(agent_list):
                idx = 0
    
    env.close()