#from .RL_utils import *
#from .RL_deep import *
#from utils.Ballgame_utils import *
import numpy as np
agbaseline=74
opbaseline=10
field_length=abs(opbaseline-agbaseline)

class Pong_funcs():

    def ball_position(self,frame):
        return ball_position_inner(frame)

    def ball_towards(self,frame,frame_prev):
        ball_pos = self.ball_position(frame)
        ball_pos_prev = self.ball_position(frame_prev)
        if (ball_pos != None) and (ball_pos_prev != None):
            towards = ball_pos[1] > ball_pos_prev[1]
        else:
            towards = False
        return towards


    def ball_in_field(self,ball_pos):
        in_field=False
        if (ball_pos !=None):
            in_field = (ball_pos[1] > opbaseline) and (ball_pos[1] < agbaseline)
        return in_field

    def crit_binary(self,s):
        frame=s[1]
        frame_prev=s[0]
        crit=0
        if self.ball_in_field(self.ball_position(frame_prev)) and self.ball_towards(frame,frame_prev):
            crit=1
        return crit

    def crit_bothdir(self,s):
        frame=s[1]
        frame_prev=s[0]
        crit=0
        if self.ball_in_field(self.ball_position(frame_prev)):
            dist_dirc=dist_directional(self.ball_position(frame_prev),self.ball_towards(frame,frame_prev) )
            crit=1-dist_dirc/(2*field_length)
        return crit

    def critfunc(self,s,crittype):
        assert (crittype in [1,2])
        if crittype==1:
            crit=self.crit_binary(s)
        elif crittype==2:
            crit=self.crit_bothdir(s)
        return crit





###################################
# functions that don't belong to the class
#####################################
def dist_to_agents_baseline(ball):
    return agbaseline-ball[1]

def dist_directional(ball,towards):
# distance that considers the direction of the ball
    if towards:
        dist=agbaseline-ball[1]
    else:
        dist=(ball[1]-opbaseline)+field_length
    return dist


def ball_position_inner(frame):
    cropped_frame=crop_screen(frame)
    rows,cols=np.where(cropped_frame==236)
    if rows.size>0:
        return rows.max(),cols.max()
    else:
        return None

def crop_screen(I):
    I = I[14:78]
    return I

