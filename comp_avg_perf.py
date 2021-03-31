import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/advice/soft_treshold'
thismodel_dir='simulations/no_advice'
#thismodel_dir='simulations/advice/unlimited'
#thismodel_dir='simulations/advice_budget/with_crit/5K'
seeds=[1,2,3,4,5]
#seeds=[4]
print('loading performance data from '+ thismodel_dir)
max_steps=2e6
wind=5
plot_flg=False
steps_pt_sum=0
min_episodes=10e6
for seed in seeds:
    with open(thismodel_dir+'/perf_'+str(seed)+'.json', 'r') as f:
        perf = json.load(f)
    min_episodes = min(min_episodes, len(perf['eval_rewards']))
sum_eval_rewards=np.zeros(min_episodes)
for seed in seeds:
    with open(thismodel_dir + '/perf_' + str(seed) + '.json', 'r') as f:
        perf = json.load(f)
    eval_rewards=np.array(perf['eval_rewards'][:min_episodes])
    sum_eval_rewards+=eval_rewards
avg_eval_rewards=sum_eval_rewards/len(seeds)
np.save(thismodel_dir+'/avg_rewards.npy',avg_eval_rewards)
'''
mypl = plt
mypl.plot(avg_eval_rewards)
mypl.savefig('avg_eval_rewards.png')
'''
a=7