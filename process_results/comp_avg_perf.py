import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='../simulations/Gopher/no_advice'
thismodel_dir='../simulations/Gopher/advice/4e-2'
seeds=[1,2,3,4,5]
#seeds=[1]
max_steps=2e6
plot_flg=False
steps_pt_sum=0
min_train_episodes,min_eval_episodes=10e6,10e6
print('averaging over {} simulations ...'.format(len(seeds)))
for seed in seeds:
    with open(thismodel_dir+'/perf_'+str(seed)+'.json', 'r') as f:
        perf = json.load(f)
    min_eval_episodes = min(min_eval_episodes, len(perf['eval_rewards']))
    min_train_episodes = min(min_train_episodes, len(perf['steps']))
print('min_eval_episodes = '+str(min_eval_episodes))

sum_eval_rewards=np.zeros(min_eval_episodes)
sum_advice_cnt=np.zeros(min_train_episodes)
for seed in seeds:
    filename=thismodel_dir + '/perf_' + str(seed) + '.json'
    print('loading '+ filename +'...')
    with open(filename, 'r') as f:
        perf = json.load(f)
    valid_eval_episodes_ratio = round(sum(perf['env_ok_eval']) / len(perf['env_ok_eval']), 2)
    print('valid_eval_episodes_ratio = ' + str(valid_eval_episodes_ratio))
    eval_rewards=np.array(perf['eval_rewards'][:min_eval_episodes])
    advice_cnt=np.array(perf['advice_cnt'][:min_train_episodes])
    sum_eval_rewards+=eval_rewards
    sum_advice_cnt+=advice_cnt
avg_eval_rewards=sum_eval_rewards/len(seeds)
avg_advice_cnt=sum_advice_cnt/len(seeds)
print('saving avg rewards ...')
np.save(thismodel_dir+'/avg_rewards.npy',avg_eval_rewards)
np.save(thismodel_dir+'/avg_advice_cnt.npy',avg_advice_cnt)

'''
mypl = plt
mypl.plot(avg_eval_rewards)
mypl.savefig('avg_eval_rewards.png')
'''
a=7