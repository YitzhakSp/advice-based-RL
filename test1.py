import numpy as np
from scipy.stats import norm
x={'a':7}
seed=6
np.random.seed(seed)
np.random.seed(12)
a='models/FRANKbootstrap_fasteranneal_pong16/FRANKbootstrap_fasteranneal_pong.pkl'
a=norm.cdf(0.05,loc=0.1,scale=0.02)

print(a)
