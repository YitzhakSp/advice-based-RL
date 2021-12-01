import json
import matplotlib.pyplot as plt
import numpy as np
eval_freq=10 #evaluation episode once every x episodes
perf_files=[('../models/gridworld/no_advice/300/avg_rewards.npy','no_adv'),
            ('../models/gridworld/with_advice/importance/1e-2/avg_rewards.npy','adv_1e-2'),
            ('../models/gridworld/with_advice/importance/2e-2/avg_rewards.npy', 'adv_2e-2'),
            ('../models/gridworld/with_advice/importance/5e-2/avg_rewards.npy', 'adv_5e-2'),
            ('../models/gridworld/with_advice/importance/1e-1/avg_rewards.npy', 'adv_1e-1'),
            ('../models/gridworld/with_advice/importance/2e-1/avg_rewards.npy', 'adv_2e-1')
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
