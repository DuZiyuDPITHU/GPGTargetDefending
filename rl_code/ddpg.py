import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys

from model import Actor, Critic
from noise_nn_ou import NormalNoise, OUNoise
from utilities import soft_update

class DDPGAgent:
    def __init__(self,
                 num_agents, 
                 obs_size, 
                 action_size, 
                 seed, 
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 theta,
                 sigma,
                 tau,
                 hidden_layers,
                 use_bn, 
                 use_reset, 
                 noise,
                 device
                ):

        """Initialize an DDPG Agent object.
        Params
        ======
            num_agents (int): number of agents
            obs_size (int): dimension of each observation
            action_size (int): dimension of each action
            seed (int): random seed
            hidden_layers (list): size of hidden_layers
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
            theta (float): parameter for noise
            sigma (float): parameter for noise
            tau (float): for soft update of target parameters
            use_bn (bool): use batch norm or not. default True
            use_reset (bool): weights initialization used in the ddpg original paper. default True
            noise (str): choose noise type, gauss(Gaussian) or OU(Ornstein-Uhlenbeck process).
        """
        self.obs_full_size = num_agents*obs_size
        self.action_full_size = num_agents*action_size
        self.seed = random.seed(seed)
        self.device = device

        # Actor Network (w/ Target Network) for decentralized execution
        self.actor_local = Actor(obs_size, 48, action_size, seed, conv_type="sage").to(device)
        self.actor_target = Actor(obs_size, 48, action_size, seed, conv_type="sage").to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network) for centralized training
        self.critic_local = Critic(self.obs_full_size, self.action_full_size, seed, hidden_layers, use_bn, use_reset).to(device)
        self.critic_target = Critic(self.obs_full_size, self.action_full_size, seed, hidden_layers, use_bn, use_reset).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Noise process
        if noise == "gauss":
            self.noise = NormalNoise(action_size, seed, mu=0., sigma=sigma)
        elif noise == "OU":
            self.noise = OUNoise(action_size, seed, theta=theta, sigma=sigma)
        else:
            sys.exit(["set noise as gauss or OU"])
        
        # Soft update
        self.tau = tau

    def act(self, observation, eps=0):
        """Returns an action for a given observation as current policy.
        
        Params
        ======
            observation (dgl.graph): a dgl graph with features
            eps (float): epsilon, for exploration action space
            return (np.array): (num_agents, action_size)
        """
        # observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        observation = observation.to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(observation, observation.ndata["feat"].float()).cpu().data.numpy()
        self.actor_local.train()

        action += eps*self.noise.sample()
        '''
        for i in range(action.shape[0]):
            if np.linalg.norm(action[i]) > 1.5:
                action[i] = action[i] / np.linalg.norm(action[i]) * 1.5
        '''
        

        return action.clip(-1.5, 1.5)

    def get_tensor_act(self, obs, target):
        """Returns actions for given observations as current policy.
        
        Params
        ======
            obs (dgl.graph): a dgl graph with features
            target (bool): actions from target or local policy network
            return (tensor): minibatch of actions of a single agent, (batch_size, action_size)
        """
        obs = obs.to(self.device)
        if target:
            with torch.no_grad():
                actions = self.actor_target(obs, obs.ndata["feat"].float())
        else:
            actions = self.actor_local(obs, obs.ndata["feat"].float())
        return actions

    def reset(self):
        self.noise.reset()

    def update_critic(self, obs_full, actions_full, next_obs_full, next_actions_full, rewards, dones, gamma):
        """Update value parameters using given batch of experiences from all agents.
        Q_targets = r + γ * critic_target(next_obs_full, next_actions_full)
        where:
            collect_actions_from(next_obs_full) -> next_actions_full
            critic_target(obs_full, actions_full) -> Q-value
        Params
        ======
            gamma (float): discount factor
        Shape
        ======
            obs_full          : (batch_size, num_agents*obs_size)
            actions_full      : (batch_size, num_agents*action_size)
            next_obs_full     : (batch_size, num_agents*obs_size)
            next_actions_full : (batch_size, num_agents*action_size)
            rewards           : (batch_size, 1)
            dones             : (batch_size, 1)
        """

        # Get target Q values from all observations and actions 
        obs_full = obs_full.to(self.device)
        actions_full = actions_full.to(self.device)
        next_obs_full = next_obs_full.to(self.device)
        next_actions_full = next_actions_full.to(self.device)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_obs_full, next_actions_full)
 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(obs_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, obs_full, actions_pred_full):
        """Update policy parameters using given batch of experiences from all agents.
        Params
        ======
            agent_number (int): detach the predicted actions from other agents to save computation of derivative
        Shape
        ======
            obs_full          : (batch_size, num_agents*obs_size)
            actions_pred_full : (batch_size, num_agents*action_size)
        """
        obs_full = obs_full.to(self.device)
        actions_pred_full = actions_pred_full.to(self.device)
        actor_loss = -self.critic_local(obs_full, actions_pred_full).mean()

        self.actor_optimizer.zero_grad()
        #actor_loss.backward()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
    
    def update_targets(self):
        soft_update(self.actor_local, self.actor_target, self.tau)
        soft_update(self.critic_local, self.critic_target, self.tau)