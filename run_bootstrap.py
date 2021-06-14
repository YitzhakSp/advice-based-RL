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
from pong_utils import *
from params import *

print('answer the upcoming questions with y or n')
long_exp=input("are you running a long experiment (1h +) ?")
assert(long_exp=='y' or long_exp=='n')
if long_exp=='y':
    print('Lets go through the checklist ... ')
    input('proper choice of device (CPU or GPU) ?')
    input('load correct model ?')
    input('compute uncertainty or not ?')
    input('use advice ?')
    input('correct advice model ?')
    input('limited advice ?')
    input('correct uncertainty treshold (type and value) ?')
    input('correct advice budget (if loading model, dont forget to adjust advbudg) ?')
    input('ask advice only in critical states ?')
    input('correct criticality type ?')
    print('finished checklist !')

load_model=False
#load_model=True
env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                  num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['seed_env'],
                  dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])
replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                             frame_height=info['NETWORK_INPUT_SIZE'][0],
                             frame_width=info['NETWORK_INPUT_SIZE'][1],
                             agent_history_length=info['HISTORY_SIZE'],
                             batch_size=info['BATCH_SIZE'],
                             num_heads=info['N_ENSEMBLE'],
                             bernoulli_probability=info['BERNOULLI_PROBABILITY'])
advice_cnt_tot=0
if load_model:
    print('loading model from: %s' %info['model_loadpath'])
    model_dict = torch.load(info['model_loadpath'])
    #info = model_dict['info']
    #info['DEVICE'] = device
    # set a new random seed
    #info["SEED"] = model_dict['cnt']
    model_base_filedir = os.path.split(info['model_loadpath'])[0]
    info['loaded_from'] = info['model_loadpath']
    perf=model_dict['perf']
    start_step_number = perf['steps'][-1]
    start_last_save=start_step_number
    advice_cnt_tot=sum(perf['advice_cnt'])
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
            'max_uncertainty':[],
            'advice_cnt':[]}
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
seed_everything(info["seed_torch_and_np"])
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
if load_model:
    # what about random states - they will be wrong now???
    # TODO - what about target net update cnt (TODO from johana)
    target_net.load_state_dict(model_dict['target_net_state_dict'])
    policy_net.load_state_dict(model_dict['policy_net_state_dict'])
    opt.load_state_dict(model_dict['optimizer'])
    print("loaded model state_dicts")
    buffer_loadpath = info['model_loadpath'].replace('.pkl', '_train_buffer.npz')
    print("auto loading buffer from:%s" %buffer_loadpath)
    try:
        replay_memory.load_buffer(buffer_loadpath)
    except Exception as e:
        print(e)
        print('not able to load from buffer: %s. exit() to continue with empty buffer' %buffer_loadpath)
advice_net=None
if info['advice_flg']:
    print('loading advice model from: %s' %info['advicemodel_loadpath'])
    model_dict = torch.load(info['advicemodel_loadpath'])
    advice_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                             n_actions=env.num_actions,
                             network_output_size=info['NETWORK_INPUT_SIZE'][0],
                             num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    if info['PRIOR']:
        advice_net = NetWithPrior(advice_net, prior_net, info['PRIOR_SCALE'])
    advice_net.load_state_dict(model_dict['policy_net_state_dict'])
action_getter = ActionGetter(n_actions=env.num_actions,
                             policy_net=policy_net,
                             random_seed=info['seed_expl'],
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
'advice_net':advice_net,
'replay_memory':replay_memory,
'opt':opt,
'model_base_filepath':model_base_filepath,
'model_base_filedir':model_base_filedir,
'env':env,
'heads':heads,
'pong_funcs_obj': Pong_funcs(),
'advice_cnt_tot':advice_cnt_tot,
'randg_adv':np.random.RandomState(info['seed_advice'])
}
train(start_step_number,
      start_last_save,
      action_getter,
      mvars,
      perf)

