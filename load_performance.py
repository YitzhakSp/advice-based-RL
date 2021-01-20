import pickle
from config import *
import json
thismodel_dir=model_savedir+'FRANKbootstrap_fasteranneal_pong02'
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf.json', 'r') as f:
    x = json.load(f)
a=7