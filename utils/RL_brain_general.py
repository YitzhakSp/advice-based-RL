
import numpy as np
import pandas as pd
#from .misc import *
from .RL_utils import *
# for tabular learning
class GenLearnAgent(object):
    def __init__(self, get_actions,q_init,to_str_funcs,gamma,q_tab=pd.DataFrame(),
                 alpha=0.1,dbg=False):
        self.dbg=dbg
        self.q_tab=q_tab
        self.q_init=q_init
        self.get_actions=get_actions
        self.alpha=alpha
        self.gamma=gamma
        self.s_to=to_str_funcs['s_to_str']
        self.a_to=to_str_funcs['a_to_str']

    def sa_to(self,s,a):
        s_str=self.s_to(s)
        a_str=self.a_to(a)
        return s_str+'_act_'+a_str


    def sa_str_sep(self,sa_str):
        spl=sa_str.split('_act_')
        return spl[0],spl[1]

    def add_state_to_qtab(self,s):
        actions_str=[self.a_to(a) for a in self.get_actions(s)]
        if self.s_to(s) not in self.q_tab.index:
            self.q_tab = self.q_tab.append(pd.Series(
                    [self.q_init] * len(actions_str),
                    index=actions_str,name=self.s_to(s)) )


    def choose_action_from_qtab(self,s,eps):
        assert(self.s_to(s) in self.q_tab.index)
        actions = self.get_actions(s)
        action_values =np.array( [self.q_tab.loc[self.s_to(s),self.a_to(a)] for a in actions])
        return choose_action(actions,action_values,eps)

    # this method really does not belong to a general learning agent, but is there for convenience
    def comp_importance(self,s):
        action_values = np.array([self.q_tab.loc[self.s_to(s), self.a_to(a)] for a in actions])
        imp=action_values.max()-action_values.min()
        return imp

class SarsaAgent(GenLearnAgent):

    def learn(self, s, a, r, s_, a_,done):
        assert(self.s_to(s) in self.q_tab.index)
        assert(self.s_to(s_) in self.q_tab.index)
        q_predict = self.q_tab.loc[self.s_to(s), self.a_to(a)]
        if not done :
            q_target = r + self.gamma * self.q_tab.loc[self.s_to(s_), self.a_to(a_)]  # next state is not terminal
        else:
            q_target=r
        error = q_target - q_predict
        if self.dbg:
            print('s=', s)
            print('update with error=', error)
        self.q_tab.loc[self.s_to(s), self.a_to(a)] += self.alpha * error


class QAgent(GenLearnAgent):

    def learn(self, s, a, r, s_,done):
        assert(self.s_to(s) in self.q_tab.index)
        assert(self.s_to(s_) in self.q_tab.index)
        q_predict = self.q_tab.loc[self.s_to(s), self.a_to(a)]
        if not done :
            q_s_ = self.q_tab.loc[self.s_to(s_)].values
            q_target = r + self.gamma * max(q_s_[~np.isnan(q_s_)])  # next state is not terminal
        else:
            q_target=r
        error = q_target - q_predict
        if self.dbg:
            print('s=', s)
            print('update with error=', error)
        self.q_tab.loc[self.s_to(s), self.a_to(a)] += self.alpha * error

class MCAgent(GenLearnAgent):
    def __init__(self, get_actions,q_init,to_str_funcs,
                 alpha=0.1, gamma=1.0, eps=0.1,dbg=False):
        super(MCAgent, self).__init__(get_actions,q_init,to_str_funcs,
                                               alpha=alpha, gamma=gamma, eps=eps,dbg=dbg)
        self.wait_list=[]


    def update_q_values(self):
        rcum=0
        for x in reversed(self.wait_list):
            rcum+=x[2]
            error = rcum - self.q_tab.loc[x[0], x[1]]
            self.q_tab.loc[x[0], x[1]] += self.alpha * error
            rcum*=self.gamma

    def learn(self, s, a, r, s_, done):
        assert (self.s_to(s) in self.q_tab.index)
        assert (self.s_to(s_) in self.q_tab.index)
        self.wait_list.append((self.s_to(s),self.a_to(a),r))
        if done:
            self.update_q_values()
            self.wait_list = []











