from utils.other_utils import *
import json
import matplotlib.pyplot as plt

avg=False
mypl=plt
mypl.ylabel('num_advice')
#mypl.title('Pong_advice')
#mypl.plot(perf['steps'],perf['advice_cnt'])
if avg:
    files = [('../simulations/advice/limited/0.04/critno/avg_advice_cnt.npy', 'BDQN-adv'),
             ('../simulations/advice/limited/0.04/crityes/avg_advice_cnt.npy', 'BDQN-crit1'),
             ('../simulations/advice/limited/0.04/critxuncert/avg_advice_cnt.npy', 'BDQN-crit2')
             ]
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
    perf_file = '../simulations/Gopher/advice/20e-2/perf_1.json'
    print("plotting advice_budget for 1 run ...")
    '''
    max_steps = 2.5e6
    mypl.xlabel('steps')
    mypl.xlim(right=max_steps)
    '''
    with open(perf_file, 'r') as f:
        perf = json.load(f)
    advice_cnt=np.array(perf['advice_cnt'])
    advice_cum=np.cumsum(advice_cnt)
    mypl.subplot(1, 2, 1)
    mypl.plot(perf['steps'],advice_cum,label='cummulative advice')
    mypl.legend()
    steps=np.array(perf['steps'])
    steps_in_ep=steps[1:]-steps[:-1]
    steps_in_ep=np.concatenate((np.array([steps[0]]),steps_in_ep))
    advice_ratio=advice_cnt/steps_in_ep
    mypl.subplot(1, 2, 2)
    mypl.yscale('log')
    mypl.plot(perf['steps'],advice_ratio,label='advice ratio')
    mypl.legend()
mypl.grid(True)
mypl.savefig('advice_consumption.png')
a=8