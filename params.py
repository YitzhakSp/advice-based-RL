import time
import datetime
cuda=True
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
print("running on %s"%device)
info = {
    #"GAME":'roms/breakout.bin', # gym prefix
    "GAME":'roms/pong.bin', # gym prefix
    "DEVICE":device, #cpu vs gpu set by argument
    "NAME":'FRANKbootstrap_fasteranneal_pong', # start files with name
    "DUELING":True, # use dueling dqn
    "DOUBLE_DQN":True, # use double dqn
    "PRIOR":True, # turn on to use randomized prior
    "PRIOR_SCALE":10, # what to scale prior by
    "N_ENSEMBLE":10, # number of bootstrap heads to use. when 1, this is a normal dqn
    "LEARN_EVERY_STEPS":4, # updates every 4 steps in osband
    "BERNOULLI_PROBABILITY": 0.9, # Probability of experience to go to each head - if 1, every experience goes to every head
    "TARGET_UPDATE":10000, # how often to update target network
    "MIN_HISTORY_TO_LEARN":500, # in steps
    "NORM_BY":255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
    'COMP_UNCERT': False,
    'UNCERT_FREQ': 1,
    "EPS_INITIAL":1.0, # should be 1
    "EPS_FINAL":0.01, # 0.01 in osband
    "EPS_EVAL":0.0, # 0 in osband, .05 in others....
    "EPS_ANNEALING_FRAMES":int(1e6), # this may have been 1e6 in osband
    #"EPS_ANNEALING_FRAMES":0, # if it annealing is zero, then it will only use the bootstrap after the first MIN_EXAMPLES_TO_LEARN steps which are random
    "EPS_FINAL_FRAME":0.01,
    "NUM_EVAL_EPISODES":1, # num examples to average in eval
    "BUFFER_SIZE":int(1e4), # Buffer size for experience replay
    "CHECKPOINT_EVERY_EPISODES":1, # how often to write pkl of model and npz of data buffer
    "EVAL_FREQUENCY":2, # how often to run evaluation episodes
    "ADAM_LEARNING_RATE":6.25e-5,
    "RMS_LEARNING_RATE": 0.00025, # according to paper = 0.00025
    "RMS_DECAY":0.95,
    "RMS_MOMENTUM":0.0,
    "RMS_EPSILON":0.00001,
    "RMS_CENTERED":True,
    "HISTORY_SIZE":4, # how many past frames to use for state input
    "BATCH_SIZE":32, # Batch size to use for learning
    "GAMMA":.99, # Gamma weight in Q update
    "PLOT_EVERY_EPISODES": 50,
    "CLIP_GRAD":5, # Gradient clipping setting
    "seed_env":101,
    'seed_expl':5, #default:122 (used before 20feb)
    'seed_torch_and_np':1234,
    "RANDOM_HEAD":-1, # just used in plotting as demarcation
    "NETWORK_INPUT_SIZE":(84,84),
    "START_TIME":time.time(),
    "MAX_STEPS":int(2.5e6), # 50e6 steps is 200e6 frames
    "MAX_EPISODE_STEPS":27000, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
    "FRAME_SKIP":4, # deterministic frame skips to match deepmind
    "MAX_NO_OP_FRAMES":30, # random number of noops applied to beginning of each episode
  #  "MAX_EPISODES":1200,
    "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life,
    "model_loadpath": 'ku',
    "advicemodel_loadpath": 'models/optimal_pong.pkl',
    'advice_flg':True,
    'seed_advice':1,
    'uncert_trh_type':'h', #for advice. values: soft(s) or hard(h)
    'uncert_trh': 0.08,  # for advice
    'uncert_trh_sigma':0.02,
    'advice_head': 0,
    'limited_advice_flg':False,
    'advice_budget':50e3,
    'advice_only_crit':False,
    'crit_trh':0.9, # for advice
    "dbg_flg":False

}
info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
info['load_time'] = datetime.date.today().ctime()
info['NORM_BY'] = float(info['NORM_BY'])
allowed_uncert_trh_types=['h','s']
assert(info['uncert_trh_type'] in allowed_uncert_trh_types)