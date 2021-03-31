import numpy as np
from scipy.stats import norm
from train_utils import *
from env import Environment
from dqn_model import EnsembleNet, NetWithPrior

#PRIOR=True
model_path='models/optimal_pong_dummy.pkl'
env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                  num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['seed_env'],
                  dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])
mvars={'env':env}
model_dict = torch.load(model_path)
info=model_dict['info']
policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                  n_actions=env.num_actions,
                                  network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                  num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
if info['PRIOR']:
    prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                            n_actions=env.num_actions,
                            network_output_size=info['NETWORK_INPUT_SIZE'][0],
                            num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    policy_net = NetWithPrior(policy_net, prior_net, info['PRIOR_SCALE'])
print('loading model from: ',model_path)
policy_net.load_state_dict(model_dict['policy_net_state_dict'])
print('model loading complete')
action_getter = ActionGetter(n_actions=env.num_actions,
                             policy_net=policy_net,
                             random_seed=1,
                             eps_initial=info['EPS_INITIAL'],
                             eps_final=info['EPS_FINAL'],
                             eps_final_frame=info['EPS_FINAL_FRAME'],
                             eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                             eps_evaluation=info['EPS_EVAL'],
                             replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                             max_steps=info['MAX_STEPS'])
evaluate(1,action_getter,mvars)