import json
import matplotlib.pyplot as plt

thismodel_dir='simulations/advice/'
print('loading performance data from '+ thismodel_dir)
with open(thismodel_dir+'/perf.json', 'r') as f:
    perf = json.load(f)
mypl=plt
mypl.xlabel('episodes')
mypl.ylabel('steps')
#mypl.title('Pong_advice')
mypl.plot(perf['steps'])
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('num_steps.png')
a=7