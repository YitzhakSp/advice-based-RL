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
from other_utils import *
import json


class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    """This class is from fg91's dqn. TODO put my function back in"""
    def __init__(self, n_actions, policy_net,eps_initial=1, eps_final=0.1, eps_final_frame=0.01,
                 eps_evaluation=0.0, eps_annealing_frames=100000,
                 replay_memory_start_size=50000, max_steps=25000000, random_seed=122):
        """
        Args:
            n_actions: Integer, number of possible actions
            eps_initial: Float, Exploration probability for the first
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames
        self.replay_memory_start_size = replay_memory_start_size
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(random_seed)
        self.policy_net=policy_net

        # Slopes and intercepts for exploration decrease
        if self.eps_annealing_frames > 0:
            self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_frames
            self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
            self.slope_2 = -(self.eps_final - self.eps_final_frame)/(self.max_steps - self.eps_annealing_frames - self.replay_memory_start_size)
            self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_steps

    def pt_get_action(self, step_number, state, active_head=None, evaluation=False):
        """
        Args:
            step_number: int number of the current step
            state: A (4, 84, 84) sequence of frames of an atari game in grayscale
            active_head: number of head to use, if None, will run all heads and vote
            evaluation: A boolean saying whether the agent is being evaluated
        Returns:
            An integer between 0 and n_actions
        """
        if evaluation:
            eps = self.eps_evaluation
        elif step_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif self.eps_annealing_frames > 0:
            # TODO check this
            if step_number >= self.replay_memory_start_size and step_number < self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope*step_number + self.intercept
            elif step_number >= self.replay_memory_start_size + self.eps_annealing_frames:
                eps = self.slope_2*step_number + self.intercept_2
        else:
            eps = 0
        if self.random_state.rand() < eps:
            return eps, self.random_state.randint(0, self.n_actions)
        else:
            state = torch.Tensor(state.astype(np.float)/info['NORM_BY'])[None,:].to(info['DEVICE'])
            vals = self.policy_net(state, active_head)
            if active_head is not None:
                action = torch.argmax(vals, dim=1).item()
                return eps, action
            else:
                # vote
                acts = [torch.argmax(vals[h],dim=1).item() for h in range(info['N_ENSEMBLE'])]
                data = Counter(acts)
                action = data.most_common(1)[0][0]
                return eps, action



def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4, plot_title=''):
    f,ax=plt.subplots(1,1,figsize=(6,6))
    for key in plot_dict.keys():
        ax.plot(rolling_average(plot_dict[key]['index']), rolling_average(plot_dict[key]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[key]['index']), rolling_average(plot_dict[key]['val']), label=key, s=3)
    ax.legend()
    if plot_title != '':
        plt.title(plot_title)
    plt.savefig(name)
    plt.close()

def matplotlib_plot_all(p,model_base_filedir):
    print('creating plots')
    episode_num = len(p['steps'])
    epochs = np.arange(episode_num)
    steps = p['steps']
    plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_step']}}, name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
    plot_dict_losses({'episode steps':{'index':epochs,'val':p['episode_relative_times']}}, name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
    plot_dict_losses({'episode head':{'index':epochs, 'val':p['episode_head']}}, name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
    plot_dict_losses({'steps loss':{'index':steps, 'val':p['episode_loss']}}, name=os.path.join(model_base_filedir, 'steps_loss.png'))
    plot_dict_losses({'steps eps':{'index':steps, 'val':p['eps_list']}}, name=os.path.join(model_base_filedir, 'steps_mean_eps.png'), rolling_length=0)
    plot_dict_losses({'steps reward':{'index':steps,'val':p['episode_reward']}},  name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
    plot_dict_losses({'episode reward':{'index':epochs, 'val':p['episode_reward']}}, name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
    plot_dict_losses({'episode times':{'index':epochs,'val':p['episode_times']}}, name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
    #plot_dict_losses({'steps avg reward':{'index':steps,'val':p['avg_rewards']}}, name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
    plot_dict_losses({'eval rewards':{'index':p['eval_steps'], 'val':p['eval_rewards']}}, name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)

def handle_checkpoint(last_save, episode_num, mvars, perf):
    if episode_num % info['CHECKPOINT_EVERY_EPISODES']==0:
        st = time.time()
        print("saving performance and model ...")
        last_save = episode_num
        model_state = {'info':info,
                 'optimizer': mvars['opt'].state_dict(),
                 'episode_num':episode_num,
                 'policy_net_state_dict':mvars['policy_net'].state_dict(),
                 'target_net_state_dict':mvars['target_net'].state_dict(),
                'perf':perf
                }

        filename = os.path.abspath(mvars['model_base_filepath'] + ".pkl")
        save_checkpoint(model_state, filename)
        with open(mvars['model_base_filedir']+'/perf.json', 'w') as fp:
            json.dump(perf, fp)
        # npz will be added
        buff_filename = os.path.abspath(mvars['model_base_filepath'] + "_train_buffer" )
        mvars['replay_memory'].save_buffer(buff_filename)
        #print("finished checkpoint", time.time()-st)
    return last_save


def ptlearn(states, actions, rewards, next_states, terminal_flags, masks,mvars):
    states = torch.Tensor(states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float)/info['NORM_BY']).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn - 50000 steps
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    mvars['opt'].zero_grad()
    q_policy_vals = mvars['policy_net'](states, None)
    next_q_target_vals = mvars['target_net'](next_states, None)
    next_q_policy_vals = mvars['policy_net'](next_states, None)
    cnt_losses = []
    for head_id in range(info['N_ENSEMBLE']):
        #TODO finish masking (this TODO is from johana)
        total_used = torch.sum(masks[:,head_id])
        if total_used > 0.0:
            next_q_vals = next_q_target_vals[head_id].data
            if info['DOUBLE_DQN']:
                next_actions = next_q_policy_vals[head_id].data.max(1, True)[1]
                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
            else:
                next_qs = next_q_vals.max(1)[0] # max returns a pair
            preds = q_policy_vals[head_id].gather(1, actions[:,None]).squeeze(1)
            targets = rewards + info['GAMMA'] * next_qs * (1-terminal_flags)
            totalloss_thishead = F.smooth_l1_loss(preds, targets, reduction='mean')
            totalloss_bysample_thishead = masks[:,head_id]*totalloss_thishead
            loss = torch.sum(totalloss_bysample_thishead/total_used)
            cnt_losses.append(loss)
            losses[head_id] = loss.cpu().detach().item()
    loss = sum(cnt_losses)/info['N_ENSEMBLE']
    loss.backward()
    for param in mvars['policy_net'].core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *=1.0/float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(mvars['policy_net'].parameters(), info['CLIP_GRAD'])
    mvars['opt'].step()
    return np.mean(losses)

def train(step_number,
          last_save,
          action_getter,
          mvars,
          perf):
    episode_num = len(perf['steps'])
    advice_cnt_tot=mvars['advice_cnt_tot']
    while episode_num < info['MAX_EPISODES']:
        print('episode '+ str(episode_num))
        terminal = False
        life_lost = True
        state = mvars['env'].reset()
        start_steps = step_number
        start_time = time.time()
        episode_reward_sum = 0
        active_head = np.random.randint(info['N_ENSEMBLE'])
        advice_cnt_thisep=0
        if info['dbg_flg']:
            advice_cnt_thisep_hard=0
            perf['advice_cnt_hard']=[]
        if ['COMP_UNCERT']:
            min_uncertainty = compute_uncertainty(state, mvars['policy_net'])
            max_uncertainty=min_uncertainty
        episode_num += 1
        ep_eps_list = []
        ptloss_list = []
        while not terminal:
            if info['COMP_UNCERT'] and step_number % info['UNCERT_FREQ']==0:
                #print('computing uncertainty ...')
                uncertainty=compute_uncertainty(state, mvars['policy_net'])
                min_uncertainty=min(uncertainty,min_uncertainty)
                max_uncertainty=max(uncertainty,max_uncertainty)
            if life_lost:
                action = 1
                eps = 0
            else:
                get_advice=False
                potential_advice_state=advice_required(state,mvars['policy_net'],info['uncert_trh_type'],mvars)
                if info['dbg_flg']:
                    advice_cnt_thisep_hard+=int(advice_required(state,mvars['policy_net'],trh_type='h',mvars=mvars))
                if info['advice_flg']:
                    if info['limited_advice_flg']:
                        if advice_cnt_tot<info['advice_budget']:
                            get_advice=potential_advice_state
                    else:
                        get_advice=potential_advice_state
                    if info['advice_only_crit']:
                        crit = mvars['pong_funcs_obj'].crit_binary(state)
                        get_advice= (get_advice and (crit>=info['crit_trh']))
                        if info['dbg_flg']:
                            if crit<info['crit_trh']:
                                print(get_advice)
                                print('no adv due to low crit')
                if get_advice:
                    #print('uncert: ',uncertainty)
                    #print('getting advice')
                    state_tens = torch.Tensor(state.astype(np.float) / info['NORM_BY'])[None, :].to(info['DEVICE'])
                    vals=mvars['advice_net'](state_tens,info['advice_head'])
                    action = torch.argmax(vals, dim=1).item()
                    advice_cnt_thisep+=1
                    advice_cnt_tot+=1
                else:
                    #print('uncert: ',uncertainty)
                    #print('no advice')
                    eps,action = action_getter.pt_get_action(step_number, state=state,active_head=active_head)
            ep_eps_list.append(eps)
            next_state, reward, life_lost, terminal = mvars['env'].step(action)
            # Store transition in the replay memory
            if info['dbg_flg']:
                '''
                frame=next_state[1]
                frame_prev=next_state[0]
                ball_position=mvars['pong_funcs_obj'].ball_position(frame)
                print('ballpos =', ball_position)
                towards=mvars['pong_funcs_obj'].ball_towards(frame,frame_prev)
                crit=mvars['pong_funcs_obj'].crit_binary(next_state)
                #print('towards agent =', towards)
                print('crit =', crit)
                time.sleep(1)
                '''
                xyz=3
            mvars['replay_memory'].add_experience(action=action,
                                            frame=next_state[-1],
                                            reward=np.sign(reward), # TODO -maybe there should be +1 here
                                            terminal=life_lost)
            step_number += 1
            episode_reward_sum += reward
            state = next_state
            if step_number % info['LEARN_EVERY_STEPS'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                #print('performing learning step')
                _states, _actions, _rewards, _next_states, _terminal_flags, _masks = mvars['replay_memory'].get_minibatch(info['BATCH_SIZE'])
                ptloss = ptlearn(_states, _actions, _rewards, _next_states, _terminal_flags, _masks,mvars)
                ptloss_list.append(ptloss)
            if step_number % info['TARGET_UPDATE'] == 0 and step_number >  info['MIN_HISTORY_TO_LEARN']:
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print('updating target network at %s'%step_number)
                mvars['target_net'].load_state_dict(mvars['policy_net'].state_dict())
        print('num advice : ',advice_cnt_thisep)
        end_time = time.time()
        ep_time = end_time-start_time
        perf['steps'].append(step_number)
        perf['episode_step'].append(step_number-start_steps)
        perf['episode_head'].append(active_head)
        perf['eps_list'].append(np.mean(ep_eps_list))
        perf['episode_loss'].append(np.mean(ptloss_list))
        perf['episode_reward'].append(episode_reward_sum)
        perf['episode_times'].append(ep_time)
        perf['episode_relative_times'].append(time.time()-info['START_TIME'])
        perf['advice_cnt'].append(advice_cnt_thisep)
        if info['dbg_flg']:
            perf['advice_cnt_hard'].append(advice_cnt_thisep_hard)
            print('advice_hard:', advice_cnt_thisep_hard)
            print('advice_soft:', advice_cnt_thisep)
        #perf['avg_rewards'].append(np.mean(perf['episode_reward'][-100:]))
        if info['COMP_UNCERT']:
            perf['min_uncertainty'].append(min_uncertainty)
            perf['max_uncertainty'].append(max_uncertainty)
        last_save = handle_checkpoint(last_save, episode_num,mvars,perf)
        if not episode_num%info['PLOT_EVERY_EPISODES'] and step_number > info['MIN_HISTORY_TO_LEARN']:
            # TODO plot title (TODO from johana)
            #print('avg reward', perf['avg_rewards'][-1])
            print('last rewards', perf['episode_reward'][-info['PLOT_EVERY_EPISODES']:])
            matplotlib_plot_all(perf,mvars['model_base_filedir'])
            '''
            with open('rewards.txt', 'a') as reward_file:
                print(len(perf['episode_reward']), step_number, perf['avg_rewards'][-1], file=reward_file)
            '''
        if episode_num%info['EVAL_FREQUENCY']==0:
            avg_eval_reward = evaluate(step_number,action_getter,mvars)
            perf['eval_rewards'].append(avg_eval_reward)
            perf['eval_steps'].append(step_number)
            matplotlib_plot_all(perf,mvars['model_base_filedir'])

def evaluate(step_number,action_getter,mvars):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    evaluate_step_number = 0
    frames_for_gif = []
    results_for_eval = []
    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        state = mvars['env'].reset()
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        while not terminal:
            if life_lost:
                action = 1
            else:
                eps,action = action_getter.pt_get_action(step_number, state, active_head=None, evaluation=True)
            next_state, reward, life_lost, terminal = mvars['env'].step(action)
            evaluate_step_number += 1
            episode_steps +=1
            episode_reward_sum += reward
            if not i:
                # only save first episode
                frames_for_gif.append(mvars['env'].ale.getScreenRGB())
                results_for_eval.append("%s, %s, %s, %s" %(action, reward, life_lost, terminal))
            state = next_state
        eval_rewards.append(episode_reward_sum)

    print("Evaluation score:\n", np.mean(eval_rewards))
    '''
    generate_gif(mvars['model_base_filedir'], step_number, frames_for_gif, eval_rewards[0], name='test', results=results_for_eval)
    # Show the evaluation score in tensorboard
    efile = os.path.join(mvars['model_base_filedir'], 'eval_rewards.txt')
    with open(efile, 'a') as eval_reward_file:
        print(step_number, np.mean(eval_rewards), file=eval_reward_file)
    '''
    return np.mean(eval_rewards)
