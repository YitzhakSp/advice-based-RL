import json
import matplotlib.pyplot as plt
import numpy as np
eval_freq=2 #evaluation episode once every x episodes
'''
files=[('../simulations/Pong/no_advice/avg_rewards_smooth.npy','BDQN-plain'),
    ('../simulations/Pong/advice/unlimited_hardtrh/0.05/avg_rewards_smooth.npy','adv 5e-2'),
       ('../simulations/Pong/advice/unlimited_hardtrh/0.10/avg_rewards_smooth.npy', 'adv 10e-2')
       ]
'''
perf_files=[('../simulations/Gopher/no_advice/avg_rewards_smooth.npy','no adv'),
            ('../simulations/Gopher/advice/15e-2/avg_rewards_smooth.npy','adv')]
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
