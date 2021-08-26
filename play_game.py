from env import Environment
from params import *
import numpy as np
random_seed=1
random_state = np.random.RandomState(random_seed)
env = Environment(rom_file='roms/ms_pacman.bin', frame_skip=info['FRAME_SKIP'],
                  num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['seed_env'],
                  dead_as_end=info['DEAD_AS_END'])
s=env.reset()
i=0
while True:
    i+=1
    action=random_state.randint(0, env.num_actions)
    next_state, reward, life_lost, terminal = env.step(action)
    if life_lost:
        print("life lost (step "+str(i)+')')
    if terminal:
        print("episode finished (step "+str(i)+')')
    print(reward)
