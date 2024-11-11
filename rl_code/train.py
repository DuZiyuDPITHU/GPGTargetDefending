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

n_episodes=1000
eps_start=1.0
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
print('-------- Model structure --------')
print('-------- Actor --------')
print(graphagent.agent.actor_local)
print('-------- Critic -------')
print(graphagent.agent.critic_local)
print('---------------------------------')   
############################################

scores_agent = []          
mean_score_list = []                                # list containing scores from each episode and agent
scores_window = deque(maxlen=100)                          # last 100 scores
step_window = deque(maxlen=100)                            # last 100 avg_steps
eps = eps_start                                            # initialize epsilon
best_score = -np.inf
is_First = True

print('Interacting with env ...')   
for i_episode in range(1, n_episodes+1):
    states = env.reset()                                    # get the current state                             
    graphagent.reset()
    done = False
    scores = np.zeros(num_agents)
    step_count = 0                           # initialize the score (for each agent)
    info = None
    while True:
        # env.render()
        step_count += 1
        actions = graphagent.act(states, eps)
        next_states, rewards, dones, info = env.step(actions)
        # print("obs feat dtype: ", states.ndata['feat'].dtype, " next_obs feat dtype: ", next_states.ndata['feat'].dtype)
        graphagent.step(states, actions, rewards, next_states, dones)
        states = next_states                               # roll over states to next time step
        scores += rewards                                  # update the score (for each agent)
        if dones == True:                                  # exit loop if episode finished
            break
    score = np.sum(scores)
    scores_window.append(score)         # save most recent score
    step_window.append(step_count)      # save most recent step
    # scores_agent.append(score)          # save most recent score
    eps = max(eps_end, eps_decay*eps)   # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.3f}, Average Step {:.3f}, Capture Rate {:.3f}, Collision Rate {:.3f}'.format(i_episode, np.mean(scores_window), np.mean(step_window), info['capture rate'], info['collision rate']), end="")
    mean_score_list.append(score)
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.3f}, Average Step {:.3f}, Capture Rate {:.3f}, Collision Rate {:.3f}'.format(i_episode, np.mean(scores_window), np.mean(step_window), info['capture rate'], info['collision rate']))

    if i_episode >=150 and np.mean(scores_window)>=goal_score and np.mean(scores_window)>=best_score:
        if is_First:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode-100, np.mean(scores_window)))
            is_First = False
        print("Save model at episode {} with rewarding {:.3f}".format(i_episode, np.mean(scores_window)))
        torch.save(graphagent.agent.actor_local.state_dict(), cp_actor + "{}.pth".format(i_episode))
        torch.save(graphagent.agent.critic_local.state_dict(), cp_critic + "{}.pth".format(i_episode))
        best_score = np.mean(scores_window)
        with open('data.pkl', 'wb') as file:
            pickle.dump(mean_score_list, file)
