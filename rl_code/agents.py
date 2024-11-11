"""
    main code that contains the neural network setup
    policy + critic updates
    see ddpg.py for other details in the network

"""

import numpy as np
import torch
import dgl
from ddpg import DDPGAgent
from utilities import ReplayBuffer_Graph

class GraphAgent:
    def __init__(self, 
                 num_players,
                 obs_size,
                 action_size, 
                 config,
                 seed,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        
        single_agent = DDPGAgent(
                 num_players, 
                 obs_size, 
                 action_size, 
                 seed = seed, 
                 lr_actor = config['LR_ACTOR'],
                 lr_critic = config['LR_CRITIC'],
                 weight_decay = config['WEIGHT_DECAY'],
                 theta = config['THETA'],
                 sigma = config['SIGMA'],
                 tau = config['TAU'],
                 hidden_layers = config['hidden_layers'],
                 use_bn = config['use_bn'], 
                 use_reset = config['use_reset'], 
                 noise = config['noise'],
                 device = device)

        # multi agents set up
        self.agent = single_agent
        
        self.num_players = num_players
        self.obs_size = obs_size
        self.action_size = action_size
        self.config = config
        self.device = device

        self.batch_size = config['BATCH_SIZE']
        self.gamma = config['GAMMA']
        self.tau = config['TAU']
        self.update_every = config['UPDATE_EVERY']

        # Replay memory
        self.memory = ReplayBuffer_Graph(config['BUFFER_SIZE'], self.batch_size, seed, device)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, obs, actions, rewards, next_obs, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        Shape
        ======
            obs (dgl.graph)
            actions (np.array):     (num_players, action_size)
            rewards (list):         (num_players,)
            next_obs (dgl.graph)
            dones (bool)
        """
        # Save experience / reward  --->  utils.py
        # print(obs.ndata['feat'].dtype)
        self.memory.add(obs, actions, rewards, next_obs, dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, obs, eps=0.):
        """Returns actions for given observations as per current policy.
        
        Params
        ======
            obs (list of dgl.graph): current obs
            eps (float): epsilon, for exploration action space
            return (np.array): actions for each agent following their policy
        """
        actions = self.agent.act(obs, eps)
        # actions = [agent.act(obs[i], eps) for i, agent in enumerate(self.agents)]
        actions = np.vstack(actions)

        return actions
    
    def collect_actions_from(self, obs, target):
        """Returns minibatch of actions for given observations as per current policy.
        
        Params
        ======
            obs (list of dgl.graph): current obs from all agents, (batch_size, graph with num_players nodes)
            target (bool): actions from target or local policy network
            return (tensor): batch of actions for each agent following their policy, (batch_size, num_players, action_size)
        """
        obs = dgl.batch(obs)
        actions = self.agent.get_tensor_act(obs, target)
        actions = actions.view(-1, self.num_players, self.action_size)
        # actions = [agent.get_tensor_act(obs[i], target) for i, agent in enumerate(self.agents)]
        # actions = torch.stack(actions)

        return actions

    def reset(self):
        self.agent.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples from all agents.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (x, a, r, x', dones) tuples 
            gamma (float): discount factor
        Shape
        ======
            x : (batch_size, num_players, obs_size)
            a : (batch_size, num_players, action_size)
            r : (batch_size, num_players)
            x': (batch_size, num_players, obs_size)
            dones : (batch_size, num_players)
        """
        obs, actions, rewards, next_obs, dones = experiences

        # reshape the size like (batch_size, num_players*action_size) for input
        actions_full = actions.reshape(self.batch_size,-1)
        next_actions_full = self.collect_actions_from(next_obs,target=True).reshape(self.batch_size,-1)  # for critic
        actions_pred_full = self.collect_actions_from(obs,target=False).reshape(self.batch_size,-1)      # for actor

        # for i, agent in enumerate(self.agents):
            # update critics
        obs_full = torch.cat([obs[i].ndata['feat'].view(1, -1) for i in range(self.batch_size)], dim=0)
        next_obs_full = torch.cat([next_obs[i].ndata['feat'].view(1, -1) for i in range(self.batch_size)], dim=0)
        self.agent.update_critic(obs_full, actions_full, next_obs_full, next_actions_full, \
                            torch.sum(rewards, dim=-1).unsqueeze(1), dones.unsqueeze(1), gamma)

        # update actors
        self.agent.update_actor(obs_full, actions_pred_full)

        # update targets
        self.agent.update_targets()