#original implementation from
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/maze.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heapq
from copy import deepcopy
from gridworld_stuff.gw_draw_utils import *

class Gridworld:
    def __init__(self,arch):

        self.compact_state_flg=True
        # all possible actions
        self.actions = ['up','down','left','right']
        self.num_actions=len(self.actions)
        self.start_agent_pos = arch['start_pos']
        self.agent_pos = self.start_agent_pos.copy()
        self.WORLD_WIDTH = arch['WORLD_WIDTH']
        self.WORLD_HEIGHT = arch['WORLD_HEIGHT']
        self.goal = arch['goal']
        self.pits = arch['pits']
        self.walls=arch['walls']
        self.poison=arch['poison']

        if self.compact_state_flg:
            self.state=self.agent_pos
        else:
            self.state=agent_pos_to_state(self.agent_pos,self.goal,self.pits)

        # max steps
        self.max_steps = float('inf')
        self.gridworld_image=GridWorldImg(self.WORLD_WIDTH,self.WORLD_HEIGHT)
    def reset(self):
        self.agent_pos = self.start_agent_pos.copy()
        if self.compact_state_flg:
            self.state=self.agent_pos
        else:
            self.state=agent_pos_to_state(self.agent_pos,self.goal,self.pits)
        return self.state.copy()

    def step(self, action):
        x, y = self.agent_pos
        life_lost, terminal = False, False
        if action == 'up':
            x_new,y_new = max(x - 1, 0),y
        elif action == 'down':
            x_new,y_new = min(x + 1, self.WORLD_HEIGHT - 1),y
        elif action == 'left':
            x_new,y_new = x,max(y - 1, 0)
        elif action == 'right':
            x_new,y_new = x,min(y + 1, self.WORLD_WIDTH - 1)
        else:
            raise Exception('undefined action')
        if [x_new, y_new] in self.pits:
            reward = -1.0
            life_lost, terminal = True, True
        elif [x_new, y_new] in self.walls:
            x_new,y_new=x,y
            reward=0.0
        elif [x_new, y_new] in self.poison:
            reward=-0.01
        elif [x_new, y_new] == self.goal:
            reward = 2.0
            life_lost, terminal = False, True
        else:
            reward = 0.0
        self.agent_pos=[x_new,y_new]
        if self.compact_state_flg:
            self.state=self.agent_pos
        else:
            self.state=agent_pos_to_state(self.agent_pos,self.goal,self.pits)
        return self.state.copy(), reward, terminal

    def get_actions(self,s):
        return self.actions

    def draw(self):
        #self.gridworld_image.annotation_add((0,0),'P')
        self.gridworld_image.circle_add((0,0))
        self.gridworld_image.tile_add((self.goal[1],self.goal[0]))
        for w in self.walls:
            self.gridworld_image.tile_add((w[1],w[0]), (0, 0, 0))
        for p in self.poison:
            self.gridworld_image.circle_add((p[1],p[0]), (100, 0, 0),radius_ratio=0.5)
        for p in self.pits:
            self.gridworld_image.circle_add((p[1], p[0]), (0, 100, 0))
        self.gridworld_image.update_screen()
        self.gridworld_image.main()





def agent_pos_to_state(pos,goal,pits):
    state=[pos[0]-goal[0],pos[1]-goal[1]]
    for pit in pits:
        state.append(pos[0]-pit[0])
        state.append(pos[1]-pit[1])
    state=np.array(state,dtype=int)
    state=np.expand_dims(state,axis=1)
    state=np.expand_dims(state,axis=0)
    return state

def position_diff(a,b):
    return [a[0]-b[0] , a[1]-b[1]]

def gw_s_to_str(s):
    return '('+str(s[0])+','+str(s[1])+')'

def gw_a_to_str(a):
    return a
