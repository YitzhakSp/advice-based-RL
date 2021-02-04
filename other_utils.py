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
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint
from params import *

def get_head_outputs_as_numpy(state,qfunc):
    state1 = np.expand_dims(state, axis=0)
    state1_tens = torch.Tensor(state1.astype(np.float) / info['NORM_BY']).to(info['DEVICE'])
    q_tens = qfunc(state1_tens, None)
    head_outputs={}
    #loop over heads
    for i,x in enumerate(q_tens):
        head_outputs['head_'+str(i)]=x.cpu().detach().numpy()
    return head_outputs

def compute_uncertainty(state, qfunc):
    heads_values = get_head_outputs_as_numpy(state, qfunc)
    num_actions=heads_values['head_0'].shape[0]
    action_values={}
    for i in range(num_actions):
        action_values['a_'+str(i)]=np.array([heads_values[key][i] for key in heads_values])
    variances_sum=0
    for key in action_values:
        variances_sum+=np.var(action_values[key])
    uncertainty=variances_sum/num_actions
    return uncertainty

def advice_required(s,qfunc,trh_type,mvars):
    uncertainty=compute_uncertainty(s,qfunc)
    adv_req=False
    if trh_type=='h':
        if uncertainty>info['uncert_trh']:
            adv_req=True
    elif trh_type=='s':
        x=mvars['randg_adv'].normal(info['uncert_trh'],info['uncert_trh_sigma'])
        if uncertainty>x:
            adv_req=True
    return adv_req

def firstind_above(mylist,val):
    for i,x in enumerate(mylist):
        if x>=val:
            return i
    return None

def rolling_average(a, n=5) :
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
