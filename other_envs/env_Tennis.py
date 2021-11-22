from gym import spaces
import copy
import numpy as np

def move_racket(racketpos,a,field_width):
    if a==0: #up
        racketpos_new=max(racketpos-1,0)
    elif a==1 : #down
        racketpos_new=min(racketpos+1,field_width-1)
    elif a==2: #stay
        racketpos_new=racketpos
    return racketpos_new


class state:
    def __init__(self,ball,velocity,me,oponnent):
        self.ball=ball
        self.velocity=velocity
        self.me=me
        self.oponnent = oponnent

    def to_str(self):
        return str(self.ball[0])+'_'+str(self.ball[1])+'_'+str(self.me)+'_'+str(self.oponnent)
    def to_img(self,width,length):
        img=np.zeros([width,length])
        img[self.ball[0],self.ball[1]]=1.0
        img[self.me,length-2]=1.0
        img[self.oponnent,1]=1.0
        return img

def s_to_str(s):
    if s is None:
        return 'none'
    else:
        return s.to_str()

class simple_pong_env:
    def __init__(self,length,width):
        self.field_length=length
        self.field_width=width
        self.action_space = spaces.Discrete(3)
        self.randprob_agent=.1
        self.randprob_oponnent=.2

    def reset(self):
        self.s=state(np.array([int(self.field_width / 2), int(self.field_length / 2)], dtype=int),
                     np.array( [np.random.randint(-1,high=2) , 1],dtype=int),
                     np.random.randint(self.field_width), np.random.randint(self.field_width))
        return copy.deepcopy(self.s)


    def step(self,a):
        assert(self.s.ball[0] >= 0 and self.s.ball[0] < self.field_width)
        assert(self.s.ball[1] >= 0 and self.s.ball[1] < self.field_length)
        r=0
        done=False
        if np.random.rand() > self.randprob_agent:
            a_real=a
        else:
            a_real = self.action_space.sample()

        self.s.me=move_racket(self.s.me, a_real, self.field_width)
        a_oponnent=self.oponnent_policy()
        self.s.oponnent=move_racket(self.s.oponnent, a_oponnent, self.field_width)
        self.s.ball+=self.s.velocity
        #if self.s.ball[1]==1 or (self.s.ball[1]==(self.field_length-2)): #ball at racketcolumn
        if self.s.ball[1]==1 and self.s.oponnent==self.s.ball[0]: #oponnent hits ball
            self.s.velocity[1]*=-1
        if self.s.ball[1]==(self.field_length-2) and self.s.me == self.s.ball[0]: #agent hits ball
            self.s.velocity[1] *= -1
        if self.s.ball[0]==0 or (self.s.ball[0]==self.field_width-1): #ball at wall
            self.s.velocity[0]*=-1
        if self.s.ball[1] == 0:
            r=1
            done=True
        if self.s.ball[1] == (self.field_length-1):
            r = -1
            done = True
        info='dummy'
        return copy.deepcopy(self.s),r,done, info

    def oponnent_policy(self):
        if np.random.rand() > self.randprob_oponnent:
            if self.s.velocity[1] == -1:
                ball_nxt = self.s.ball + self.s.velocity
                if ball_nxt[0]>self.s.oponnent:
                    a_op=1
                elif ball_nxt[0]<self.s.oponnent:
                    a_op=0
                else:
                    a_op=2
            else:
                a_op = 2
        else:
            a_op = self.action_space.sample()
        return a_op




