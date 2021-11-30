import json
import matplotlib.pyplot as plt
import numpy as np
eval_freq=10 #evaluation episode once every x episodes
perf_files=[('../models/gridworld/300/avg_rewards.npy','300'),
            ('../models/gridworld/400/avg_rewards.npy','400')]
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
