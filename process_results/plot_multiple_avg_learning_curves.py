import json
import matplotlib.pyplot as plt
import numpy as np
eval_freq=10 #evaluation episode once every x episodes
adv_budg='200'
perf_files = [('../simulations/Gridworld/with_advice/arch_3/importance/adv_budget_'+adv_budg+'/2e-2/rewards_statistics.npy', 'imp_2e-2'),
              ('../simulations/Gridworld/with_advice/arch_3/importance/adv_budget_'+adv_budg+'/1e-2/rewards_statistics.npy', 'imp_1e-2'),
              ('../simulations/Gridworld/with_advice/arch_3/crit/adv_budget_'+adv_budg+'/rewards_statistics.npy', 'crit'),
              ('../simulations/Gridworld/no_advice/arch_3/rewards_statistics.npy', 'no_adv')
              ]
plt.xlabel('episode')
plt.ylabel('reward')
z=1.96
for f in perf_files:
    reward_stats=np.load(f[0])
    mean = reward_stats[0]
    mrg_of_er=z*reward_stats[1]
    x=np.arange(0,eval_freq*mean.shape[0],eval_freq)
    plt.plot(x,mean,label=f[1])
    plt.fill_between(x,mean-mrg_of_er,mean+mrg_of_er,alpha=0.1)
    plt.legend()
plt.grid(True)
print('saving rewards plot ...')
plt.savefig('rewards.png')
