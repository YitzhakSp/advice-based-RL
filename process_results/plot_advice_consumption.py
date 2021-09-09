import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/no_advice'
files=[('../simulations/advice/limited/0.04/critno/avg_advice_cnt.npy','BDQN-adv'),
('../simulations/advice/limited/0.04/crityes/avg_advice_cnt.npy','BDQN-crit1'),
('../simulations/advice/limited/0.04/critxuncert/avg_advice_cnt.npy','BDQN-crit2')
]
avg=True
'''
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf_3.json', 'r') as f:
    perf = json.load(f)
'''
mypl=plt
mypl.ylabel('num_advice')
#mypl.title('Pong_advice')
#mypl.plot(perf['steps'],perf['advice_cnt'])
if avg:
    mypl.xlabel('episodes')
    mypl.xlim(0, 1070)
    for f in files:
        print("plotting average advice_budg ...")
        avg_advice_cnt = np.load(f[0])
        advice_cum=np.cumsum(avg_advice_cnt)
        x=range(len(advice_cum))
        mypl.plot(x,advice_cum,label=f[1])
        #mypl.axhline(150000, linestyle='--')
        plt.legend()
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
mypl.savefig('../plots/num_advice.png')
a=7