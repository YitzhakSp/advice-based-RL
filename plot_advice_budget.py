import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

thismodel_dir='simulations/advice/'
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf.json', 'r') as f:
    perf = json.load(f)
mypl=plt
mypl.xlabel('episodes')
mypl.ylabel('num_advice')
#mypl.title('Pong_advice')
mypl.plot(perf['advice_cnt'])
advice_cum=np.cumsum(np.array(perf['advice_cnt']))
mypl.plot(advice_cum)
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('num_advice.png')
a=7