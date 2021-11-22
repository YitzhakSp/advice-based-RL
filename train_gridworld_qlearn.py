#import gym
import numpy
import random
import pandas
import json
#from utils.misc import *
import matplotlib.pyplot as plt
from utils.RL_brain_general import *
from gridworld_stuff.gridworld import *

eps=.1
gamma=0.9
num_episodes=200
env = Gridworld()
to_str_funcs={
's_to_str':gw_s_to_str,
'a_to_str':gw_a_to_str
}
scores = []
q_init=-100
#s_term=[(0,'g'),(length-1,'g')]
ag = QAgent(env.get_actions, q_init, to_str_funcs,eps=eps)
for episode in range(num_episodes):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('episode ' + str(episode))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    s=env.reset()
    ag.add_state_to_qtab(s)
    step=0
    dfact=1.0
    score=0
    while True:
        a = ag.choose_action_from_qtab(s,eps)
        s_, r, terminal = env.step(a)
        ag.add_state_to_qtab(s_)
        score+=dfact*r
        ag.learn(s, a, r, s_, terminal)
        s=s_
        step+=1
        dfact*=gamma
        if terminal:
            print('score:{}'.format(score))
            break

with open('scores.json', 'w') as fp:
    json.dump(scores, fp)
plt.plot(scores)
plt.show()
print('scores={}'.format(scores))
