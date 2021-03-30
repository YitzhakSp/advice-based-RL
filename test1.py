import numpy as np
from scipy.stats import norm
x={'a':7}
seed=6
np.random.seed(seed)
np.random.seed(12)
a='models/FRANKbootstrap_fasteranneal_pong16/FRANKbootstrap_fasteranneal_pong.pkl'
a=norm.cdf(0.05,loc=0.1,scale=0.02)

print(a)
lot_flg=False
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
    steps_pt=perf['eval_steps'][ind] # s