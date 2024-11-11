import torch
import numpy as np
import gym
import gym_confrontation_game
from agents import GraphAgent

# 加载环境
env = gym.make('maconfrontation-v0')
env = env.unwrapped

# 配置参数
config = {
    'BUFFER_SIZE': int(1e6),         # replay buffer size
    'BATCH_SIZE' : 256,              # minibatch size
    'GAMMA' : 0.99,                  # discount factor
    'TAU' :1e-3,                     # for soft update of target parameters
    'LR_ACTOR' : 1e-3,               # learning rate of the actor
    'LR_CRITIC' : 1e-3,              # learning rate of the critic
    'WEIGHT_DECAY' : 0,              # L2 weight decay
    'UPDATE_EVERY' : 1,              # how often to update the network
    'THETA' : 0.15,                  # parameter for Ornstein-Uhlenbeck process
    'SIGMA' : 0.2,                   # parameter for Ornstein-Uhlenbeck process and Gaussian noise
    'hidden_layers' : [256,128],     # size of hidden_layers
    'use_bn' : True,                 # use batch norm or not 
    'use_reset' : True,              # weights initialization used in original ddpg paper
    'noise' : "gauss"                # choose noise type, gauss(Gaussian) or OU(Ornstein-Uhlenbeck process) 
}

# 初始化代理
num_agents = env.n_agents
state_size = env.n_features
action_size = env.nu
graphagent = GraphAgent(num_agents, state_size, action_size, config, seed=0)

# 加载训练好的模型权重
graphagent.agent.actor_local.load_state_dict(torch.load("cp_actor_from_agent_240.pth"))
graphagent.agent.critic_local.load_state_dict(torch.load("cp_critic_from_agent_240.pth"))
Collision_rates = []
Capture_rates = []
# 测试网络推理表现
test_episodes = 10
for episode in range(test_episodes):
    state = env.reset()
    done = False
    total_reward = np.zeros(num_agents)
    while not done:
        # env.render()
        action = graphagent.act(state, eps=0.3)  # 在测试时不使用epsilon-greedy策略
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        if done == True:
            break
    Collision_rates.append(info['collision rate'])
    Capture_rates.append(info['capture rate'])
    print(f"Episode {episode + 1}: Total Reward: {np.mean(total_reward)}, Collision rate: {info['collision rate']}, Capture rate: {info['capture rate']}")
print("All test episodes are done. Average Collision rate: {:.2f}%, Average Capture rate: {:.2f}%".format(np.mean(Collision_rates)*100, np.mean(Capture_rates)*100))
env.close()