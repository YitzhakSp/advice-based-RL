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

if not dbg:
    print('Lets go through the checklist ...')
    if use_advice:
        input('correct qfunc for advice ?')
random_seeds=[1]
env = Gridworld(start_agent_pos=[4,5])
gw_to_str_funcs={
's_to_str':gw_s_to_str,
'a_to_str':gw_a_to_str
}
if dbg:
    advice_required=True
if use_advice:
    advice_ag = QAgent(get_actions=env.get_actions,
                       q_init=q_init,
                       to_str_funcs=gw_to_str_funcs,
                       q_tab=pd.read_pickle(models_dir+'/qfunc_opt.pkl'))
for current_seed in random_seeds:
    np.random.seed(current_seed)
    scores = []
    #s_term=[(0,'g'),(length-1,'g')]
    ag = QAgent(get_actions=env.get_actions,
                q_init=q_init,
                to_str_funcs=gw_to_str_funcs)
    perf={'train_steps':[],
    'eval_steps':[],
    'train_scores':[],
    'eval_scores':[]
    }
    best_score_train,best_score_eval=None,None
    steps_train = 0
    for episode in range(num_episodes):
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
        print('eps: '+str(eps))
        steps_thisep=0
        dfact=1.0
        score_train=0
        terminal=False
        while (not terminal) and (steps_thisep<max_episode_steps):
            if use_advice and advice_required:
                a = advice_ag.choose_action_from_qtab(s=s,eps=0)
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
        print('score_train: {}'.format(score_train))
        scores.append(score_train)
        perf['train_steps'].append(steps_train)
        perf['train_scores'].append(score_train)
        if best_score_train is None:
            best_score_train=score_train
        else:
            best_score_train=max(best_score_train, score_train)
        print('best score_train: {}'.format(best_score_train))
        if episode%eval_freq==0:
            print('     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('             evaluation episode ' )
            print('     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            score_eval=evaluate(ag,env,max_episode_steps,gamma)
            print('score_eval: {}'.format(score_eval))
            perf['eval_steps'].append(steps_train)
            perf['eval_scores'].append(score_eval)
            performance_improved=False
            if best_score_eval is None:
                best_score_eval = score_eval
                performance_improved=True
            else:
                if score_eval > best_score_eval:
                    best_score_eval = score_eval
                    performance_improved = True
            print('best score_eval: {}'.format(best_score_eval))
            if performance_improved:
                ag.q_tab.to_pickle(models_dir+'/qfunc.pkl')

        if episode%save_freq==0:
            with open(models_dir+'/perf_'+str(current_seed)+'.json', 'w') as fp:
                json.dump(scores, fp)

