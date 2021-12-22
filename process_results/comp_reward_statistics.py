import numpy as np

from utils.other_utils import *
import json

#thismodel_dir='../simulations/Gopher/no_advice'
thismodel_dir='../simulations/Gridworld/with_advice/arch_3/importance/adv_limit_10/2e-2'
#thismodel_dir='../simulations/Gridworld/no_advice/arch_3'

seeds=range(1,101)
#seeds=[1]
analyze_advice_cnt=True
max_steps=2e6
plot_flg=False
steps_pt_sum=0
min_train_episodes,min_eval_episodes=10e6,10e6
print('averaging over {} simulations ...'.format(len(seeds)))
for seed in seeds:
    with open(thismodel_dir+'/perf_'+str(seed)+'.json', 'r') as f:
        perf = json.load(f)
    min_eval_episodes = min(min_eval_episodes, len(perf['eval_scores']))
    min_train_episodes = min(min_train_episodes, len(perf['train_scores']))
print('min_eval_episodes = '+str(min_eval_episodes))

sum_eval_rewards=np.zeros(min_eval_episodes)
sum_advice_cnt=np.zeros(min_train_episodes)
all_rewards=np.empty([len(seeds),min_eval_episodes])
i=0
for seed in seeds:
    filename=thismodel_dir + '/perf_' + str(seed) + '.json'
    print('loading '+ filename +'...')
    with open(filename, 'r') as f:
        perf = json.load(f)
    valid_eval_episodes_ratio = round(sum(perf['env_ok_eval']) / len(perf['env_ok_eval']), 2)
    print('valid_eval_episodes_ratio = ' + str(valid_eval_episodes_ratio))
    assert(valid_eval_episodes_ratio>0.99) # if not, code needs to be changed
    all_rewards[i]=np.array(perf['eval_scores'][:min_eval_episodes])
    i+=1
    if analyze_advice_cnt:
        advice_cnt=np.array(perf['advice_cnt'][:min_train_episodes])
        sum_advice_cnt+=advice_cnt

avg_eval_rewards=np.mean(all_rewards,axis=0)
rewards_se=np.std(all_rewards,axis=0)/np.sqrt(len(seeds))
rewards_stats=np.concatenate([avg_eval_rewards[np.newaxis,...],rewards_se[np.newaxis,...]],axis=0)
print('saving avg rewards ...')
np.save(thismodel_dir+'/rewards_statistics.npy',rewards_stats)
if analyze_advice_cnt:
    avg_advice_cnt = sum_advice_cnt / len(seeds)
    np.save(thismodel_dir+'/avg_advice_cnt.npy',avg_advice_cnt)

'''
mypl = plt
mypl.plot(avg_eval_rewards)
mypl.savefig('avg_eval_rewards.png')
'''
a=7