import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/advice/soft_treshold'
thismodel_dir='simulations/no_advice'
thismodel_dir='simulations/advice/unlimited'
#thismodel_dir='simulations/advice_budget/with_crit/5K'
seeds=[1,2,3,4,5]
#seeds=[4]
print('loading performance data from '+ thismodel_dir)
max_steps=2e6
wind=5
plot_flg=False
steps_pt_sum=0
for seed in seeds:
    with open(thismodel_dir+'/perf_'+str(seed)+'.json', 'r') as f:
        perf = json.load(f)
    rewards_smooth=rolling_average(perf['eval_rewards'],n=wind)
    rewards_smooth=np.concatenate([np.array((wind-1)*[rewards_smooth[0]]),rewards_smooth])
    mypl=plt
    mypl.xlim(right=max_steps)
    mypl.xlabel('steps')
    mypl.ylabel('reward')
    #mypl.title('Pong_advice')
    mypl.plot(perf['eval_steps'],rewards_smooth)
    ind=firstind_above(rewards_smooth,9.99)
    steps_pt=perf['eval_steps'][ind] # steps until performance treshold
    steps_pt_sum+=steps_pt
    print('steps_pt_{}: {}'.format(seed,steps_pt))
    if plot_flg:
        mypl.axvline(x= steps_pt, color = 'r')
        mypl.grid(True)
        # attention: always remember this:
        mypl.show()
        mypl.savefig('perfplot_1.png')
steps_pt_avg=steps_pt_sum/len(seeds)
print('steps_pt_avg: ',steps_pt_avg )
a=7