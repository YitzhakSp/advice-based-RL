import numpy as np
import random

def choose_action(actions,action_values,eps):
    perm = np.random.permutation(len(action_values))
    action_values_perm=action_values[perm]
    actions_perm=[actions[i] for i in perm]
    opt_ind = np.argmax(action_values_perm)
    actions_nonopt=[actions_perm[i] for i in perm if i!=opt_ind]
    if np.random.rand() <= eps and len(actions_nonopt)>0:
        a = random.choice(actions_nonopt)
    else:
        a = actions_perm[opt_ind]
    return a

def comp_trg(rcum, disc_fct, q_trg, game_over):
    if not game_over:
        trg=rcum+disc_fct*max(q_trg)
    else:
        trg=rcum
    return trg

def get_epsilon(curr_game, random_frames, epsilon_start, epsilon_end, epsilon_decay_time):
    if curr_game<random_frames:
        return 1.0
    elif curr_game > epsilon_decay_time:
        return epsilon_end
    return epsilon_start + (epsilon_end - epsilon_start) * float(curr_game) / epsilon_decay_time