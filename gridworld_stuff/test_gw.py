# https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter08/maze.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq
from copy import deepcopy
from gridworld import *

gw=Gridworld()
