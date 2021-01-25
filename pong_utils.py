#from .RL_utils import *
#from .RL_deep import *
#from utils.Ballgame_utils import *
import numpy as np

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
            in_field = (ball_pos[1] > 10) and (ball_pos[1] < 74)
        return in_field

    def crit_binary(self,s):
        frame=s[1]
        frame_prev=s[0]
        crit=0
        if self.ball_in_field(self.ball_position(frame_prev)) and self.ball_towards(frame,frame_prev):
            crit=1
        return crit

###################################
# functions that don't belong to the class
#####################################
def dist_to_agents_baseline(ball):
    return agbaseline-ball[1]

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

