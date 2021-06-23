import json
import matplotlib.pyplot as plt
import numpy as np
eval_freq=2 #evaluation episode once every x episodes
files=[#('simulations/no_advice/avg_rewards_smooth.npy','no_adv'),
    ('../simulations/advice/limited/0.04/crityes/avg_rewards_smooth.npy','crityes'),
      ('../simulations/advice/limited/0.04/critno/avg_rewards_smooth.npy','critno'),
      ('../simulations/advice/limited/0.04/critxuncert/avg_rewards_smooth.npy','critxuncert'),
]
for f in files:
    y = np.load(f[0])
    x=np.arange(0,eval_freq*len(y),eval_freq)
    plt.plot(x,y,label=f[1])
    plt.legend()
print('saving rewards plot ...')
plt.savefig('../plots/rewards.png')
