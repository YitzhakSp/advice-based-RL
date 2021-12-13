import json
import matplotlib.pyplot as plt
import numpy as np
eval_freq=10 #evaluation episode once every x episodes
perf_files = [('../simulations/Gridworld/with_advice/arch_3/importance/adv_limit_100/2e-2/avg_rewards.npy', 'imp_2e-2'),
              ('../simulations/Gridworld/with_advice/arch_3/importance/adv_limit_100/1e-2/avg_rewards.npy', 'imp_1e-2'),
              ('../simulations/Gridworld/with_advice/arch_3/crit/adv_limit_100/avg_rewards.npy', 'crit'),
              ('../simulations/Gridworld/no_advice/arch_3/avg_rewards.npy', 'no_adv')
              ]
plt.xlabel('episode')
plt.ylabel('reward')
for f in perf_files:
    y = np.load(f[0])
    x=np.arange(0,eval_freq*len(y),eval_freq)
    plt.plot(x,y,label=f[1])
    plt.legend()
plt.grid(True)
print('saving rewards plot ...')
plt.savefig('rewards.png')
