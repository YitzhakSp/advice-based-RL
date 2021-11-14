#original implementation from
# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/maze.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heapq
from copy import deepcopy

class Gridworld:
    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 6

        # maze height
        self.WORLD_HEIGHT = 5

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.num_actions=len(self.actions)

        # start state
        self.start_agent_pos = [4,5]
        self.agent_pos = self.start_agent_pos.copy()

        # goal agent_pos
        self.goal = [0, 2]

        # all obstacles
        self.pits = [[0,0 ],[1,0],[2,0],[1,4],[2,5],[4,2]]
        self.state=agent_pos_to_state(self.agent_pos,self.goal,self.pits)

        # max steps
        self.max_steps = float('inf')

    def reset(self):
        self.agent_pos = self.start_agent_pos.copy()
        return self.state.copy()

    def step(self, action):
        x, y = self.agent_pos
        life_lost, terminal = False, False
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.pits:
            reward = -1.0
            life_lost, terminal = True, True
        elif [x, y] == self.goal:
            reward = 1.0
            life_lost, terminal = False, True
        else:
            reward = 0.0
        self.agent_pos=[x,y]
        self.state=agent_pos_to_state(self.agent_pos,self.goal,self.pits)
        return self.state.copy(), reward, terminal


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
