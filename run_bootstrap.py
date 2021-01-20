from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
#from IPython import embed
from collections import Counter
import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint
from env import Environment
from replay import ReplayMemory
import config
from argparse import ArgumentParser
from train_utils import *
from params import *

load_model=False
env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                  num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                  dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])
replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                             frame_height=info['NETWORK_INPUT_SIZE'][0],
                             frame_width=info['NETWORK_INPUT_SIZE'][1],
                             agent_history_length=info['HISTORY_SIZE'],
                             batch_size=info['BATCH_SIZE'],
                             num_heads=info['N_ENSEMBLE'],
                             bernoulli_probability=info['BERNOULLI_PROBABILITY'])
if load_model:
    # load data from loadpath - save model load for later. we need some of
    # these parameters to setup other things
    print('loading model from: %s' %info['model_loadpath'])
    model_dict = torch.load(info['model_loadpath'])
    info = model_dict['info']
    info['DEVICE'] = device
    # set a new random seed
    info["SEED"] = model_dict['cnt']
    model_base_filedir = os.path.split(info.model_loadpath)[0]
    start_step_number = start_last_save = model_dict['cnt']
    info['loaded_from'] = info.model_loadpath
    perf = model_dict['perf']
    start_step_number = perf['steps'][-1]
else:
    perf = {'steps':[],
            'avg_rewards':[],
            'episode_step':[],
            'episode_head':[],
            'eps_list':[],
            'episode_loss':[],
            'episode_reward':[],
            'episode_times':[],
            'episode_relative_times':[],
            'eval_rewards':[],
            'eval_steps':[],
            'min_uncertainty':[],
            'max_uncertainty':[]}
    start_step_number = 0
    start_last_save = 0
    # make new directory for this run in the case that there is already a
    # project with this name
    run_num = 0
    model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
    while os.path.exists(model_base_filedir):
        run_num +=1
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d'%run_num)
    assert(run_num<20)     #make sure that models dir doesn't get too large
    os.makedirs(model_base_filedir)
    print("----------------------------------------------")
    print("starting NEW project: %s"%model_base_filedir)
model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
write_info_file(info, model_base_filepath, start_step_number)
heads = list(range(info['N_ENSEMBLE']))
seed_everything(info["SEED"])
policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                  n_actions=env.num_actions,
                                  network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                  num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                  n_actions=env.num_actions,
                                  network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                  num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
if info['PRIOR']:
    prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                            n_actions=env.num_actions,
                            network_output_size=info['NETWORK_INPUT_SIZE'][0],
                            num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    print("using randomized prior")
    policy_net = NetWithPrior(policy_net, prior_net, info['PRIOR_SCALE'])
    target_net = NetWithPrior(target_net, prior_net, info['PRIOR_SCALE'])
target_net.load_state_dict(policy_net.state_dict())
opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])
if info['model_loadpath'] is not '':
    # what about random states - they will be wrong now???
    # TODO - what about target net update cnt
    target_net.load_state_dict(model_dict['target_net_state_dict'])
    policy_net.load_state_dict(model_dict['policy_net_state_dict'])
    opt.load_state_dict(model_dict['optimizer'])
    print("loaded model state_dicts")
    if info['buffer_loadpath'] == '':
        info['buffer_loadpath'] = info['model_loadpath'].replace('.pkl', '_train_buffer.npz')
        print("auto loading buffer from:%s" %info['buffer_loadpath'])
        try:
            replay_memory.load_buffer(info['buffer_loadpath'])
        except Exception as e:
            print(e)
            print('not able to load from buffer: %s. exit() to continue with empty buffer' %info['buffer_loadpath'])
action_getter = ActionGetter(n_actions=env.num_actions,
                             policy_net=policy_net,
                             eps_initial=info['EPS_INITIAL'],
                             eps_final=info['EPS_FINAL'],
                             eps_final_frame=info['EPS_FINAL_FRAME'],
                             eps_annealing_frames=info['EPS_ANNEALING_FRAMES'],
                             eps_evaluation=info['EPS_EVAL'],
                             replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                             max_steps=info['MAX_STEPS'])
mvars={
'policy_net':policy_net,
'target_net':target_net,
'replay_memory':replay_memory,
'opt':opt,
'model_base_filepath':model_base_filepath,
'model_base_filedir':model_base_filedir,
'env':env,
'heads':heads
}
train(start_step_number,
      start_last_save,
      action_getter,
      mvars,
      perf)

