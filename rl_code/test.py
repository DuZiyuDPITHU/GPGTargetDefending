import numpy as np
from collections import deque
import pickle
import torch
from agents import GraphAgent
import gym
import gym_confrontation_game

"""
Params
======
    n_episodes (int): maximum number of training episodes
    eps_start (float): starting value of epsilon, for exploration action space
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    goal_score (float): average score to be required
    env_file_name (str): your path to Tennis.app 
    config (dict): parameter set for training
"""

test_episodes=15
eps_start=0.01
eps_end=0.01
eps_decay=0.996
goal_score=-260.0

cp_actor="cp_actor_from_agent_"
cp_critic="cp_critic_from_agent_"
scores_file="scores_maddpg.txt"

env = gym.make('maconfrontation-v0')
env = env.unwrapped

config = {
    'BUFFER_SIZE': int(1e6),         # replay buffer size
    'BATCH_SIZE' : 256,              # minibatch size
    'GAMMA' : 1.01,                  # discount factor
    'TAU' :1e-3,                     # for soft update of target parameters
    'LR_ACTOR' : 3e-4,               # learning rate of the actor
    'LR_CRITIC' : 3e-4,              # learning rate of the critic
    'WEIGHT_DECAY' : 0,              # L2 weight decay
    'UPDATE_EVERY' : 1,              # how often to update the network
    'THETA' : 0.15,                  # parameter for Ornstein-Uhlenbeck process
    'SIGMA' : 0.2,                   # parameter for Ornstein-Uhlenbeck process and Gaussian noise
    'hidden_layers' : [256,128],     # size of hidden_layers
    'use_bn' : True,                 # use batch norm or not 
    'use_reset' : True,              # weights initialization used in original ddpg paper
    'noise' : "gauss"                # choose noise type, gauss(Gaussian) or OU(Ornstein-Uhlenbeck process) 
}

# number of agents
num_agents = env.n_agents
print('Number of agents:', num_agents)

# size of each action
action_size = env.nu
print('Size of each action:', action_size)

# examine the state space 
state_size = env.n_features
print('There are {} agents. Each observes a state with length: {}'.format(num_agents, state_size))
#######################################


###########  Multi Agent Setting  ##########
graphagent = GraphAgent(num_agents, state_size, action_size, config, seed=0)
graphagent.agent.actor_local.load_state_dict(torch.load("cp_actor_from_agent_2.pth"))
# graphagent.agent.critic_local.load_state_dict(torch.load("cp_critic_from_agent_2.pth"))
print('-------- Model structure --------')
print('-------- Actor --------')
print(graphagent.agent.actor_local)
print('-------- Critic -------')
print(graphagent.agent.critic_local)
print('---------------------------------')   
############################################

scores_agent = []
Collision_rates = []
Capture_rates = []     
eps = eps_start                                            # initialize epsilon
best_score = -np.inf
is_First = True

print('Interacting with env ...')   
for i_episode in range(1, test_episodes+1):
    states = env.reset()                                    # get the current state                             
    graphagent.reset()
    done = False
    scores = np.zeros(num_agents)
    step_count = 0                           # initialize the score (for each agent)
    info = None
    while True:
        env.render()
        step_count += 1
        actions = graphagent.act(states, eps)
        next_states, rewards, dones, info = env.step(actions)
        # print("obs feat dtype: ", states.ndata['feat'].dtype, " next_obs feat dtype: ", next_states.ndata['feat'].dtype)
        # graphagent.step(states, actions, rewards, next_states, dones)
        states = next_states                               # roll over states to next time step
        scores += rewards                                  # update the score (for each agent)
        if dones == True:                                  # exit loop if episode finished
            break
    score = np.sum(scores)
    eps = max(eps_end, eps_decay*eps)   # decrease epsilon
    Collision_rates.append(info['collision rate'])
    Capture_rates.append(info['capture rate'])
    print('\rEpisode {}\tTotal Score: {:.3f}, Step {:.3f}, Capture Rate {:.3f}, Collision Rate {:.3f}'.format(i_episode, score, step_count, info['capture rate'], info['collision rate']), end="")
print("All test episodes are done. Average Collision rate: {:.2f}%, Average Capture rate: {:.2f}%".format(np.mean(Collision_rates)*100, np.mean(Capture_rates)*100))

