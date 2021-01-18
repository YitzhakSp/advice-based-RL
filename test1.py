import numpy as np
x={'a':7}
seed=6
np.random.seed(seed)
np.random.seed(12)

for i in range(10):
    print(np.random.randint(10))

