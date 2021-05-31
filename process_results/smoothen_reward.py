import json
import matplotlib.pyplot as plt
from other_utils import *
import numpy as np
wind=5
thismodel_dir='simulations/advice/limited/0.04/critno'
#thismodel_dir='simulations/advice/unlimited_hardtrh/0.04'
#thismodel_dir='simulations/no_advice'
avg_rewards=np.load(thismodel_dir+'/avg_rewards.npy')
rewards_smooth=rolling_average(avg_rewards,n=wind)
rewards_smooth=np.concatenate([np.array((wind-1)*[rewards_smooth[0]]),rewards_smooth])
print('saving smoothened rewards ... ')
np.save(thismodel_dir+'/avg_rewards_smooth.npy',rewards_smooth)

a=7