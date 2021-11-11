import time
import datetime
#params for gridworld
cuda=True
if cuda:
    device = 'cuda'
else:
    device = 'cpu'
print("running on %s"%device)
info = {
    'print_stepnum':False,
    'printstepnum_freq':100,
    "DEVICE":device, #cpu vs gpu set by argument
    "NAME":'gw_model', # start files with name
    "DUELING":True, # use dueling dqn
    "DOUBLE_DQN":True, # use double dqn
    "PRIOR":True, # turn on to use randomized prior
    "PRIOR_SCALE":10, # what to scale prior by
    "N_ENSEMBLE":5, # number of bootstrap heads to use. when 1, this is a normal dqn
    "LEARN_EVERY_STEPS":4, # updates every 4 steps in osband
    "BERNOULLI_PROBABILITY": 0.9, # Probability of experience to go to each head - if 1, every experience goes to every head
    "TARGET_UPDATE":10000, # how often to update target network
    "MIN_HISTORY_TO_LEARN":50, # in steps
    "NORM_BY":5.0,  # divide the float(of uint) by this number to normalize - max val of data is 255
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
    "HISTORY_SIZE":1, # how many past frames to use for state input
    "BATCH_SIZE":30, # Batch size to use for learning
    "GAMMA":.9, # Gamma weight in Q update
    "APPLY_GAMMA_TO_RET":True, # apply gamma for computing episode return
    "PLOT_EVERY_EPISODES": 50,
    "CLIP_GRAD":5, # Gradient clipping setting
    "seed_env":101,
    'seed_expl':1,
    'seed_torch_and_np':1234,
    "RANDOM_HEAD":-1, # just used in plotting as demarcation
    "NETWORK_INPUT_SIZE":(14,1),
    "START_TIME":time.time(),
    "MAX_STEPS":int(5e6), # 50e6 steps is 200e6 frames
     "MAX_EPISODE_STEPS_TRAIN":200, # Orig dqn give 18k steps, Rainbow seems to give 27k steps
    "MAX_EPISODE_STEPS_EVAL": 100,  # Orig dqn give 18k steps, Rainbow seems to give 27k steps
    "FRAME_SKIP":4, # deterministic frame skips to match deepmind
  #  "MAX_EPISODES":1200,
    "DEAD_AS_END":True, # do you send finished=true to agent while training when it loses a life,
    "model_loadpath": 'models/gopher_model02/ku.pkl',
    "advicemodel_loadpath": 'models/optimal_model_gopher.pkl',
    'advice_flg':False,
    'seed_advice':1,
    'env_check_freq': 1e12, # check that environment functions correctly every n steps
    'uncert_trh_type':'h', #for advice. values: soft(s) or hard(h)
    'uncert_trh': 15e-2,  # for advice
    'uncert_trh_sigma':0.02,
    'advice_head': 0,
    'limited_advice_flg':False,
    'advice_budget':150e3,
    'crit_how':1, # 1: nocrit, 2: crit_and_uncert, 3: crit*uncert
    'crittype':2, # (1,binary), (2,bothdir)
    'crit_trh':0.9, # for advice
    "dbg_flg":False

}
info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
info['load_time'] = datetime.date.today().ctime()
info['NORM_BY'] = float(info['NORM_BY'])
allowed_uncert_trh_types=['h','s']
assert(info['uncert_trh_type'] in allowed_uncert_trh_types)