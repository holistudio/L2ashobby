import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time

from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

def train(epochs=50, steps_per_epoch=4000, max_ep_len=1000):
    env = todo()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    agent = todo()

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = agent.PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
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
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = agent.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    print(f'Episode Reward={ep_ret}, Episode Length={ep_len}')
                o, ep_ret, ep_len = env.reset(), 0, 0
                if type(o) == tuple:
                    o = o[0]

if __name__ == '__main__':
    train()