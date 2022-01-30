#import gym
import numpy
import random
import pandas
import json
#from utils.misc import *
import matplotlib.pyplot as plt
from utils.RL_brain_general import *
from utils.train_utils_qlearn import *
from gridworld_stuff.gridworld import *
from params_gridworld import *
from gridworld_stuff.arch_longwall_3 import *
import json

if not dbg:
    print('Lets go through the checklist ...')
    if use_advice:
        input('correct qfunc for advice ?')
with open(models_dir+'/info.json','w') as fp:
    json.dump(info,fp)
#random_seeds=[1,2,3,4,5,6,7,8,9,10]
random_seeds=range(11,101)
env = Gridworld(arch)
env.draw()
gw_to_str_funcs={
's_to_str':gw_s_to_str,
'a_to_str':gw_a_to_str
}
if dbg:
    advice_required=True
if use_advice:
    advice_ag = QAgent(get_actions=env.get_actions,
                       q_init=q_init,
                       gamma=gamma,
                       to_str_funcs=gw_to_str_funcs,
                       q_tab=pd.read_pickle(qfunc_adv_file))
for current_seed in random_seeds:
    print('learning procedure with seed: '+str(current_seed))
    print('in progress ...')
    np.random.seed(current_seed)
    scores = []
    #s_term=[(0,'g'),(length-1,'g')]
    ag = QAgent(get_actions=env.get_actions,
                q_init=q_init,
                gamma=gamma,
                to_str_funcs=gw_to_str_funcs)
    perf={'train_steps':[],
    'eval_steps':[],
    'eval_episodes':[],
    'train_scores':[],
    'eval_scores':[],
    'env_ok_eval':[],
    'advice_cnt':[]
    }
    best_score_train,best_score_eval=None,None
    steps_train = 0
    last_improved_eval_episode=0
    advice_cnt = 0
    for episode in range(num_episodes):
        if verbose:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('episode ' + str(episode))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        s=env.reset()
        ag.add_state_to_qtab(s)
        if episode>eps_decay_duration:
            eps=eps_final
        else:
            lambdaa=episode/eps_decay_duration
            eps=(1-lambdaa)*eps_start + lambdaa*eps_final
        steps_thisep=0
        dfact=1.0
        score_train=0
        terminal=False
        while (not terminal) and (steps_thisep<max_episode_steps):
            advice_required=False
            if use_advice:
                if limited_advice and advice_cnt>advice_budget:
                    advice_required=False
                else:
                    if advice_criterion=='imp':
                        imp=ag.comp_importance(s)
                        if imp > imp_trh:
                            advice_required=True
                    elif advice_criterion=='crit':
                        crit=comp_crit(env,s)
                        if crit > crit_trh:
                            advice_required=True
                    else:
                        raise Exception('unknown advice criterion')

            if use_advice and advice_required:
                #print('asking for advice ...')
                #print('advice_cnt ='+str(advice_cnt))
                a = advice_ag.choose_action_from_qtab(s=s,eps=0)
                advice_cnt+=1
            else:
                a = ag.choose_action_from_qtab(s,eps)
            s_, r, terminal = env.step(a)
            ag.add_state_to_qtab(s_)
            score_train+= dfact * r
            ag.learn(s, a, r, s_, terminal)
            s=s_
            steps_thisep+=1
            dfact*=gamma
        steps_train+=steps_thisep
        score_train=round(score_train, 3)
        scores.append(score_train)
        perf['train_steps'].append(steps_train)
        perf['train_scores'].append(score_train)
        perf['advice_cnt'].append(advice_cnt)
        if best_score_train is None:
            best_score_train=score_train
        else:
            best_score_train=max(best_score_train, score_train)
        if verbose:
            print('eps: ' + str(eps))
            print('score_train: {}'.format(score_train))
            print('best score_train: {}'.format(best_score_train))
        if episode%eval_freq==0:
            score_eval=evaluate(ag,env,max_episode_steps,gamma)
            perf['eval_steps'].append(steps_train)
            perf['eval_episodes'].append(episode)
            perf['eval_scores'].append(score_eval)
            perf['env_ok_eval'].append(True)
            performance_improved=False
            if best_score_eval is None:
                best_score_eval = score_eval
                performance_improved=True
            else:
                if score_eval > best_score_eval:
                    best_score_eval = score_eval
                    performance_improved = True
                    last_improved_eval_episode=episode
            if verbose:
                print('     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('             evaluation episode ')
                print('     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('score_eval: {}'.format(score_eval))
                print('best score_eval: {}'.format(best_score_eval))
                print('last improved eval episode: {}'.format(last_improved_eval_episode))
            if save_model and performance_improved:
                ag.q_tab.to_pickle(models_dir+'/qfunc.pkl')

        if episode%save_freq==0:
            with open(models_dir+'/perf_'+str(current_seed)+'.json', 'w') as fp:
                json.dump(perf, fp)
    print('last improved eval episode: {}'.format(last_improved_eval_episode))
    print('best score_eval: {}'.format(best_score_eval))


