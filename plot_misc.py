import pickle
from config import *
from other_utils import *
import json
import matplotlib.pyplot as plt

thismodel_dir='simulations/advice_budget/plain/40K'
thismodel_dir='simulations/no_advice'
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf.json', 'r') as f:
    perf = json.load(f)
max_steps=2e6
mypl=plt
mypl.xlim(right=max_steps)
mypl.ylim(0.0,0.2)
mypl.xlabel('steps')
#mypl.title('Pong_advice')
mypl.plot(perf['steps'],perf['min_uncertainty'])
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('mypl.png')
a=7