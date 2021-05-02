import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/no_advice'
thismodel_dir='simulations/advice/unlimited_hardtrh/0.06'
avg=True
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf_3.json', 'r') as f:
    perf = json.load(f)
mypl=plt
mypl.ylabel('num_advice')
#mypl.title('Pong_advice')
#mypl.plot(perf['steps'],perf['advice_cnt'])
if avg:
    print("plotting average advice_budg ...")
    mypl.xlabel('episodes')
    avg_advice_cnt = np.load(thismodel_dir+'/avg_advice_cnt.npy')
    advice_cum=np.cumsum(avg_advice_cnt)
else:
    print("plotting advice_budget for 1 run ...")
    max_steps = 2.5e6
    mypl.xlabel('steps')
    mypl.xlim(right=max_steps)
    with open(thismodel_dir+'/perf_3.json', 'r') as f:
        perf = json.load(f)
    advice_cum=np.cumsum(np.array(perf['advice_cnt']))
    mypl.plot(perf['steps'],advice_cum)
mypl.grid(True)
mypl.savefig('num_advice.png')
a=7