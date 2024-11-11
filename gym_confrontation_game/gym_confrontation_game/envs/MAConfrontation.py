import gym
import torch
import numpy as np
import configparser
import os.path as path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from sklearn.neighbors import NearestNeighbors
from gym import spaces
from gym import spaces, error, utils
from gym.utils import seeding
import random

from gym_confrontation_game.envs.Grid import Grid
from gym_confrontation_game.envs.AttackerDefender1v0 import AttackerDefender1v0
from gym_confrontation_game.envs.utilities import attackers_control_1v0, next_positions, distance, build_graph

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
matplotlib.use('TkAgg')

class ConfrontationEnv(gym.Env):

    def __init__(self):
        print("Start init confrontation env")
        config_file = path.join(path.dirname(__file__), "MAConfrontation.cfg")
        config = configparser.ConfigParser()
        with open(config_file, 'r', encoding='utf-8') as f:
            config.read_file(f)
        config = config['TESTING']

        self.dynamic = True # if the agents are moving or not
        self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not

        self.obs_num = int(config['nei_num'])
        # number states per agent
        self.nx_system = 2
        # numer of features per agent: [attacker position in observation range, self state, onehot embedding of agent status]
        self.n_features = 3*self.obs_num + self.nx_system + 2
        # number of actions per agent
        self.nu = 2 

        # problem parameters from file
        self.n_agents = int(config['n_agent'])
        self.n_atkr = int(config['n_atkr']) 

        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.cap_radius = float(config['capture_radius'])
        self.col_radius = float(config['collision_radius'])
        self.init_distribution_dist = float(config['init_distribution_dist'])
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_max_atkr = 1.0

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))
        self.feats = np.zeros((self.n_agents,self.n_features))
        self.current_attackers = [(0.0, 0.0) for _ in range(self.n_atkr)]
        self.current_attackers = [(0.0, 0.0) for _ in range(self.n_agents)]
        self.attackers_status = [0 for _ in range(self.n_atkr)]
        self.attackers_arrived= [0 for _ in range(self.n_atkr)]
        self.defenders_status = [0 for _ in range(self.n_agents)]
        self.new_arrivals = [0 for _ in range(self.n_atkr)]
        self.new_collisions = [0 for _ in range(self.n_agents)]
        self.new_captures = [0 for _ in range(self.n_atkr)]
        self.nearest_attacker_label = [-1 for _ in range(self.n_agents)]
        self.last_relative_dist = [0 for _ in range(self.n_agents)]
        self.curr_relative_dist = [0 for _ in range(self.n_agents)] 
        self.attackers_stop_index = []
        self.defenders_stop_index = []
        
        # initialize the action and observation space
        self.action_space = spaces.Box(low=-self.v_max, high=self.v_max, shape=(2 *self.n_agents,),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        # init optimal control policy for attackers
        # Initialize H-J value functions and grids
        self.value1v0 = np.load(path.join(path.dirname(__file__), '1v0AttackDefend.npy'))
        self.grid1v0 = Grid(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2, np.array([100, 100])) 
        self.tau1v0 = np.arange(start=0, stop=2.5 + 1e-5, step=0.025)
        self.agents_1v0 = AttackerDefender1v0(uMode="min", dMode="max")

        self.fig = None
        self.line1 = None
        self.line2 = None
        self.square1 = None
        self.counter = 0 
        self.seed(10)
        np.random.seed(5)
        print("Initializing confrontation env completed")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # control attackers
        control_attackers = attackers_control_1v0(self.agents_1v0, self.grid1v0, self.value1v0, self.tau1v0, self.current_attackers, self.attackers_stop_index)
        for i in range(self.n_atkr):
            if i in self.attackers_stop_index:
                control_attackers[i] = (0, 0)
        self.current_attackers = next_positions(self.current_attackers, control_attackers, self.dt)
        # control agents
        self.u = np.reshape(action,(self.n_agents, self.nu))
        for i in range(self.n_agents):
            if i in self.defenders_stop_index:
                self.u[i, :] = (0, 0)
        self.x[:, 0] = np.clip(self.x[:, 0] + self.u[:, 0]*self.dt, -1, 1)
        self.x[:, 1] = np.clip(self.x[:, 1] + self.u[:, 1]*self.dt, -1, 1)

        self.current_defenders = [np.array(self.x[i, 0:2]) for i in range(self.x.shape[0])]
        # check arrival, capture and collision
        arrived = self.attackers_arrived.copy()
        for i in range(self.n_atkr):
            if (0.6<=self.current_attackers[i][0]) and (self.current_attackers[i][0]<=0.8):
                if (0.1<=self.current_attackers[i][1]) and (self.current_attackers[i][1]<=0.3):
                    arrived[i] = 1
        self.new_arrivals = [arrived[i] - self.attackers_arrived[i] for i in range(self.n_atkr)]
        self.attackers_arrived = arrived

        captured = self.attackers_status.copy()
        collide = self.defenders_status.copy()
        for j in range(len(self.current_defenders)):
            if collide[j] == 1:
                continue

            for k in range(j+1, len(self.current_defenders)):
                if distance(self.current_defenders[j], self.current_defenders[k]) <= self.col_radius:
                    if collide[j] == 0 and collide[k] == 0:
                        collide[j] = 1
                        collide[k] = 1
                        # print(f"The defender{j} has collided with the defender{k}! \n")
            for i in range(len(self.current_attackers)):
                if (distance(self.current_defenders[j], self.current_attackers[i]) <= self.cap_radius) and (collide[j] == 0):
                    if (captured[i] == 0) and (self.attackers_arrived[i] == 0):
                        captured[i] = 1
        self.new_captures = [captured[i] - self.attackers_status[i] for i in range(self.n_atkr)]
        self.new_collisions = [collide[i] - self.defenders_status[i] for i in range(self.n_agents)]
        self.attackers_status = captured
        self.defenders_status = collide
        # update stopped attackers and defenders
        new_atkr_stop_index = self.attackers_stop_index.copy()
        for i, capture in enumerate(self.attackers_status):
            if capture and (i not in new_atkr_stop_index):
                new_atkr_stop_index.append(i)
        for j, arrived in enumerate(self.attackers_arrived):
            if arrived and (j not in new_atkr_stop_index):
                new_atkr_stop_index.append(j)
        self.attackers_stop_index = sorted(new_atkr_stop_index)
        new_def_stop_index = []
        for i, collision in enumerate(self.defenders_status):
            if collision:
                new_def_stop_index.append(i)
        self.defenders_stop_index = sorted(new_def_stop_index)
        # update done
        info = {}
        done = False 
        # print(len(self.attackers_stop_index), " : ",self.attackers_stop_index)
        if self.counter > 400 :
            done = True 
            info['collision rate'] = sum(self.defenders_status)/self.n_agents
            info['capture rate'] = sum(self.attackers_status)/self.n_atkr   
        if len(self.defenders_stop_index)==self.n_agents or len(self.attackers_stop_index)==self.n_atkr:
            done = True 
            info['collision rate'] = sum(self.defenders_status)/self.n_agents
            info['capture rate'] = sum(self.attackers_status)/self.n_atkr

        return self._get_obs(), self.instant_cost(), done, info

    def instant_cost(self):  # sum of differences in velocities
        rew_list = []
        
        for i in range(self.n_agents):
            rew_i = 0.0
            '''
            if i in self.defenders_stop_index:
                rew_i = -0.3
            elif self.nearest_attacker_label[i] != -1:
                dist = distance(self.current_defenders[i], self.current_attackers[self.nearest_attacker_label[i]])
                if dist <= self.comm_radius:
                    rew_i = -dist
            # else:
            #     rew_i = -0.15
            # rew_i *= 5
            #rew_i -= self.defenders_status[i]
            # rew_i += np.sum(np.array(self.attackers_status))/self.n_agents
            # rew_i -= self.current_defenders[i][0] - 0.9 if abs(self.current_defenders[i][0]) > 0.9 else 0
            # rew_i -= self.current_defenders[i][1] - 0.9 if abs(self.current_defenders[i][1]) > 0.9 else 0
            rew_i -= 0.5*self.new_collisions[i]
            rew_i += 5*sum(self.new_captures)/self.n_agents
            rew_i -= 0.5*sum(self.new_arrivals)/self.n_agents
            '''
            if i in self.defenders_stop_index:
                rew_i = -0.15
            else:
                rew_i = -0.5*self.curr_relative_dist[i]
            if i in self.defenders_stop_index:
                rew_i = -0.15
            elif self.nearest_attacker_label[i] != -1:
                dist = 0.5*distance(self.current_defenders[i], self.current_attackers[self.nearest_attacker_label[i]])
                if dist <= self.comm_radius:
                    rew_i = -dist
            rew_i *= 0.8
            rew_i -= 3*self.new_collisions[i]
            rew_i += 6*sum(self.new_captures)/self.n_agents
            # rew_i -= 0.5*sum(self.new_arrivals)/self.n_agents

            rew_list.append(rew_i)
        # print(np.mean(self.curr_relative_dist))
        return rew_list


    def reset(self):
        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))
        self.feats = np.zeros((self.n_agents,self.n_features))
        self.current_attackers = [(0.0, 0.0) for _ in range(self.n_atkr)]
        self.current_attackers = [(0.0, 0.0) for _ in range(self.n_agents)]
        self.attackers_status = [0 for _ in range(self.n_atkr)]
        self.attackers_arrived= [0 for _ in range(self.n_atkr)]
        self.defenders_status = [0 for _ in range(self.n_agents)]
        self.new_arrivals = [0 for _ in range(self.n_atkr)]
        self.new_collisions = [0 for _ in range(self.n_agents)]
        self.new_captures = [0 for _ in range(self.n_atkr)]
        self.nearest_attacker_label = [-1 for _ in range(self.n_agents)]
        self.last_relative_dist = [0 for _ in range(self.n_agents)]
        self.curr_relative_dist = [0 for _ in range(self.n_agents)]        
        self.attackers_stop_index = []
        self.defenders_stop_index = []
        self.counter = 0

        initials = []
        while len(initials) < self.n_atkr + self.n_agents:
            x = random.uniform(-0.8, 0.3)
            y = random.uniform(-0.8, 0.8)
            valid = True
            for player in initials:
                if ((x - player[0])**2 + (y - player[1])**2) < self.init_distribution_dist**2:
                    valid = False
                    break
            if valid:
                initials.append((x, y))

        random_set = list(set(random.sample(range(self.n_atkr+self.n_agents), self.n_atkr)))
        complement_set = list(set(range(self.n_atkr+self.n_agents)) - set(random_set))
        self.current_attackers = [initials[i] for i in random_set]
        self.current_defenders = [initials[i] for i in complement_set]
        self.x[:, 0:2] = np.array(self.current_defenders)
        return self._get_obs()

    def _get_obs(self):
        features = []
        for i in range(self.n_agents):
            if i in self.defenders_stop_index:
                features.append(torch.tensor([0.0]*self.n_features))
                self.nearest_attacker_label[i] = -1
                continue
            defender_x = self.current_defenders[i][0]
            defender_y = self.current_defenders[i][1]
            
            # sort attackers by relative distance, attacker info: (index, distance, relative_x, relative_y)
            distances = []
            for j in range(self.n_atkr): 
                if j in self.attackers_stop_index:
                    continue
                attacker_x = self.current_attackers[j][0]
                attacker_y = self.current_attackers[j][1]
                
                relative_x = attacker_x - defender_x
                relative_y = attacker_y - defender_y
                distance = np.sqrt(relative_x ** 2 + relative_y ** 2)
                distances.append((j, distance, relative_x, relative_y))

            distances = [attacker for attacker in distances if attacker[1] <= self.comm_radius]
            distances.sort(key=lambda x: x[1])
            if len(distances) == 0:
                self.last_relative_dist[i] = self.curr_relative_dist[i]
                self.curr_relative_dist[i] = 0.3
            else:
                self.last_relative_dist[i] = self.curr_relative_dist[i]
                self.curr_relative_dist[i] = np.mean([attacker[1] for attacker in distances])
            mean_obs_dist = 0.3
            if len(distances) > 0:
                mean_obs_dist = np.mean([attacker[1] for attacker in distances])
            
            if len(distances) >= self.obs_num:
                nearest_attackers = distances[:self.obs_num]
            else:
                nearest_attackers = distances + [(-1, 1, 0, 0)] * (self.obs_num - len(distances))

            # construct features
            self.nearest_attacker_label[i] = nearest_attackers[0][0]
            self_pos_feature = [defender_x, defender_y]
            self_status_feature = [0, 1] if i in self.defenders_stop_index else [1, 0]
            pos_feature = []
            for attacker in nearest_attackers:
                pos_feature.extend([mean_obs_dist, attacker[2], attacker[3]])
            features.append(torch.tensor(self_pos_feature + pos_feature + self_status_feature))
        g = build_graph(self.current_defenders, features, self.defenders_stop_index, 0.3)
        g.ndata['feat'] = torch.tensor(g.ndata['feat'], dtype=torch.float32)
        return g


    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        attacker_pos = np.array(self.current_attackers)
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            line2, = ax.plot(attacker_pos[:, 0], attacker_pos[:, 1], 'ro')
            
            #ax.plot([0], [0], 'kx')

            plt.ylim(-1.0, 1.0)
            plt.xlim(-1.0, 1.0)
            
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1
            self.line2 = line2
            

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])
        self.line2.set_xdata(attacker_pos[:, 0])
        self.line2.set_ydata(attacker_pos[:, 1])
        self.square1 = plt.Rectangle(xy=(0.6, 0.1), width=0.2, height=0.2, angle=0.0)

        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass