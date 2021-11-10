from env import Environment
from params_atari import *
import numpy as np
from PIL import Image as pil_image
from gridworld_stuff.gridworld import *
import time

random_seed=1
random_state = np.random.RandomState(random_seed)
env = Gridworld()
s=env.reset()
i=0
while True:
    i+=1
    print('state: '+str(env.state))
    action=random_state.randint(0, env.num_actions)
    next_state, reward, life_lost, terminal = env.step(action)
    #frame=next_state[0]
    #frame=pil_image.fromarray(next_state[0])
    if life_lost:
        print("life lost (step "+str(i)+')')
    if terminal:
        print("episode finished (step "+str(i)+')')
        s = env.reset()
        i=0
    print(reward)
    time.sleep(0.5)
