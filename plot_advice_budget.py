import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

thismodel_dir='simulations/advice_budget/plain/40K'
thismodel_dir='simulations/advice/soft_treshold'
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf.json', 'r') as f:
    perf = json.load(f)
max_steps=2e6
mypl=plt
mypl.xlim(right=max_steps)
mypl.xlabel('steps')
mypl.ylabel('num_advice')
#mypl.title('Pong_advice')
mypl.plot(perf['steps'],perf['advice_cnt'])
advice_cum=np.cumsum(np.array(perf['advice_cnt']))
mypl.plot(perf['steps'],advice_cum)
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('num_advice.png')
a=7