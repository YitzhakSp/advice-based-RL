#import gym
import numpy
import random
import pandas
import json
#from utils.misc import *
import matplotlib.pyplot as plt
from utils.RL_brain_general import *
from gridworld_stuff.gridworld import *

eps_start=1.0
eps_final=0.01
eps_decay_duration=1000
gamma=0.9
num_episodes=2500
max_episode_steps=100
q_init=0
env = Gridworld(start_agent_pos=[4,5])
gw_to_str_funcs={
's_to_str':gw_s_to_str,
'a_to_str':gw_a_to_str
}
scores = []
#s_term=[(0,'g'),(length-1,'g')]
ag = QAgent(get_actions=env.get_actions,
            q_init=q_init,
            to_str_funcs=gw_to_str_funcs)
best_score=None
for episode in range(num_episodes):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('episode ' + str(episode))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    s=env.reset()
    ag.add_state_to_qtab(s)
    if episode>eps_decay_duration:
        eps=eps_final
    else:
        lambdaa=episode/eps_decay_duration
        eps=(1-lambdaa)*eps_start + lambdaa*eps_final
    print('eps: '+str(eps))
    steps_thisep=0
    dfact=1.0
    score=0
    terminal=False
    while (not terminal) :
        a = ag.choose_action_from_qtab(s,eps)
        s_, r, terminal = env.step(a)
        ag.add_state_to_qtab(s_)
        score+=dfact*r
        ag.learn(s, a, r, s_, terminal)
        s=s_
        steps_thisep+=1
        dfact*=gamma
    score=round(score,3)
    print('score: {}'.format(score))
    scores.append(score)
    if best_score is None:
        best_score=score
    else:
        best_score=max(best_score,score)
    print('best score: {}'.format(best_score))

with open('scores.json', 'w') as fp:
    json.dump(scores, fp)
plt.plot(scores)
plt.show()
print('scores={}'.format(scores))
