import numpy as np
import scipy.signal

import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.optim import Adam

from utils.mpi_pytorch import mpi_avg_grads
from utils.mpi_tools import mpi_avg, mpi_statistics_scalar

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# A more robust Residual Block inspired by modern CNN architectures (e.g., ResNet)
class ResidualBlock(nn.Module):
    def __init__(self, channels, activation=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = activation()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.activation(out)
    
# A deeper CNN architecture inspired by MuZero's network design.
class CNN(nn.Module):
    def __init__(self, activation=nn.ReLU, num_res_blocks=8, res_channels=64):
        super().__init__()

        # Initial convolution to increase channels and capture basic features
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=res_channels, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(res_channels)
        self.activation = activation()

        # A "tower" of residual blocks to build deep features
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(res_channels, activation) for _ in range(num_res_blocks)]
        )

        # The output from the residual tower will be flattened.
        # For a (1, 21, 10) input, the conv output is (res_channels, 21, 10)
        # because of padding.
        self.flattened_size = res_channels * 21 * 10

    def forward(self, x):
        # The CNN body (representation function in MuZero terms)
        features = self.activation(self.initial_bn(self.initial_conv(x)))
        features = self.residual_tower(features)
        features = features.view(-1, self.flattened_size)
        return features
    
class CNNCategoricalActor(nn.Module):
    def __init__(self, body_net, action_dim, activation=nn.ReLU):
        super().__init__()
        self.body = body_net
        self.policy_head = nn.Sequential(
            nn.Linear(self.body.flattened_size, 256),
            activation(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

    def _distribution(self, obs):
        features = self.body(obs)
        logits = self.policy_head(features)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
class CNNCritic(nn.Module):

    def __init__(self, body_net, activation=nn.ReLU):
        super().__init__()
        self.body = body_net
        self.value_head = nn.Sequential(
            nn.Linear(self.body.flattened_size, 256),
            activation(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        features = self.body(obs)
        value = self.value_head(features)
        return torch.squeeze(value, -1) # Critical to ensure v has right shape.
    
class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, activation=nn.Tanh, device=None):
        super().__init__()

        obs_dim = 21*10 # observation_space.shape[0] 
        self.device = device

        # Create a single shared CNN body
        body_net = CNN(activation=activation)

        # policy builder 
        self.pi = CNNCategoricalActor(body_net=body_net, action_dim=action_space.n, activation=activation)
        # build value function
        self.v  = CNNCritic(body_net=body_net, activation=activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a

    def act(self, obs):
        return self.step(obs)[0]
    

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros(combined_shape(size, act_dim), dtype=torch.float32)
        self.adv_buf = torch.zeros(size, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.ret_buf = torch.zeros(size, dtype=torch.float32)
        self.val_buf = torch.zeros(size, dtype=torch.float32)
        self.logp_buf = torch.zeros(size, dtype=torch.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.full = False

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        if self.ptr >= self.max_size:
            self.ptr = 0 # if the buffer is full, overwrite earlier buffer records
            self.full = True
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat((self.rew_buf[path_slice], torch.tensor([last_val])))
        vals = torch.cat((self.val_buf[path_slice], torch.tensor([last_val])))
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = torch.from_numpy(discount_cumsum(deltas.numpy(), self.gamma * self.lam).copy())
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = torch.from_numpy(discount_cumsum(rews.numpy(), self.gamma)[:-1].copy())
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        if self.ptr >= self.max_size:
            self.full = True
        assert self.full   # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf.numpy())
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return data

class PPOAgent(object):
    def __init__(self, observation_space, action_space, activation=nn.Tanh,
                 local_steps_per_epoch=4096, gamma=0.99, lam=0.95, device=None,
                 clip_ratio=0.1, train_pi_iters=10, train_v_iters=10, pi_lr=3e-4, vf_lr=1e-3, target_kl=0.015):
        
        obs_dim = observation_space.shape
        act_dim = action_space.shape

        self.device = device

        self.buffer = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        self.cnn_ac = CNNActorCritic(observation_space, action_space, activation, device)
        if self.device:
            self.cnn_ac.to(self.device)

        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.cnn_ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.cnn_ac.v.parameters(), lr=vf_lr)

        self.target_kl = target_kl
        pass
    
    def focus_obs(self, obs):
        batch_size = obs.shape[0]

        # Reshape the grid part of the observation, preserving the batch dimension.
        grid = obs[:,:200].view(batch_size, 20, 10)

        # Extract the piece, row, and column information.
        current_piece = obs[:,206:206+7]
        current_row = obs[:,202:203] # Slice to keep dim: (batch_size, 1)
        current_col = obs[:,203:204] # Slice to keep dim: (batch_size, 1)

        # Create a padding tensor that matches the batch size.
        padding = torch.zeros((batch_size, 1), dtype=torch.float32, device=obs.device)

        # Construct the additional row and add a dimension for concatenation.
        add_grid_row = torch.cat((current_piece, current_row, current_col, padding), dim=-1).unsqueeze(1)

        # Concatenate the new row to the grid and add the channel dimension for the CNN.
        out = torch.cat((grid, add_grid_row), dim=1).unsqueeze(1)
        return out
    
    def step(self, obs):
        out = self.focus_obs(obs)
        return self.cnn_ac.step(out)
    
    def act(self, obs):
        out = self.focus_obs(obs)
        return self.cnn_ac.act(out)
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        out = self.focus_obs(obs)
        pi, logp = self.cnn_ac.pi(out, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        out = self.focus_obs(obs)
        return ((self.cnn_ac.v(out) - ret)**2).mean()
    
    def update(self):
        data = self.buffer.get()

        # data = self.buffer.get()
        data = {k: v.to(self.device) for k, v in data.items()}

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                # print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(self.cnn_ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.cnn_ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()
