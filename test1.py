from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import *
import time
import matplotlib.pyplot as plt


maze = RandomBlockMazeGenerator(maze_size=8, obstacle_ratio=0.0)
init_and_goal_states={'i':[1,2],'g':[[4,4]]}
env = MazeEnv(maze,init_and_goal_states=init_and_goal_states)
#env = MazeEnv(maze,goal_states=[[6,5]])
env.reset()
for i in range(100):
    print(i)
    #fig=env.render()
    a=env.action_space.sample()
    s_view, r, done, info = env.step(a)
    s=env.state
    print('state=',s)
    plt.imshow(s_view, cmap=env.cmap, norm=env.norm)
    plt.show()
    #time.sleep(1)
