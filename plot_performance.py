import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

thismodel_dir='simulations/advice/hard_treshold'
#thismodel_dir='simulations/no_advice'
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf.json', 'r') as f:
    perf = json.load(f)
max_steps=2e6
wind=5
rewards_smooth=rolling_average(perf['eval_rewards'],n=wind)
rewards_smooth=np.concatenate([np.array((wind-1)*[rewards_smooth[0]]),rewards_smooth])
mypl=plt
mypl.xlim(right=max_steps)
mypl.xlabel('steps')
mypl.ylabel('reward')
#mypl.title('Pong_advice')
mypl.plot(perf['eval_steps'],rewards_smooth)
ind=firstind_above(rewards_smooth,10)
steps_mp=perf['eval_steps'][ind] # steps until machine performance
mypl.axvline(x= steps_mp, color = 'r')
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('perfplot.png')
a=7