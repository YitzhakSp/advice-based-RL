#import gym
import numpy
import random
import pandas
import json
#from utils.misc import *
import matplotlib.pyplot as plt
from utils.RL_brain_general import *
from other_envs.random_walk import *

eps=.1
length=50
first_double_state=30
n_double_states=5
env=RWEnv(length,
         first_double_state,
         n_double_states)
to_str_funcs={
's_to_str':s_to_str,
'a_to_str':a_to_str
}
s0=(25,'sng')
scores = []
q_init=-100
#s_term=[(0,'g'),(length-1,'g')]
ag = QAgent(env.get_actions, q_init, to_str_funcs,eps=eps)

for i_episode in range(200):
    env.set_state(s0)
    s=s0
    ag.add_state_to_qtab(s)
    t=0
    while True:
        #print('t={}'.format(t))
        a = ag.choose_action_from_qtab(s,eps)
        s_, r, done = env.step(a)
        ag.add_state_to_qtab(s_)
        ag.learn(s,a,r,s_,done)
        s=s_
        t+=1
        if done:
            print('episode:{}, score:{}'.format(i_episode,t))
            scores.append(int(t))
            break

with open('scores.json', 'w') as fp:
    json.dump(scores, fp)
plt.plot(scores)
plt.show()
print('scores={}'.format(scores))
