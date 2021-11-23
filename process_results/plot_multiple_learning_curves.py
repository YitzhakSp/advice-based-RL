from utils.other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/advice/soft_treshold'

perf_files=[('../simulations/Gopher/no_advice/perf_2.json','no adv'),
    ('../simulations/Gopher/advice/10e-2/perf_1.json','10e-2'),
            ('../simulations/Gopher/advice/20e-2/perf_1.json','20e-2')]
'''
perf_files=[('../simulations/Pong/no_advice/perf_2.json','no adv'),
    ('../simulations/Pong/advice/unlimited_hardtrh/0.04/perf_2.json','4e-2'),
            ('../simulations/Pong/advice/unlimited_hardtrh/0.10/perf_2.json', '10e-2')]
'''
dbg_flg=True
max_steps_flg=False
max_steps=2e6
wind=100 #(5 for Pong, 100 for Gopher)
plot_flg=True
steps_pt_sum=0
mypl=plt
if max_steps_flg:
    mypl.xlim(right=max_steps)
mypl.xlabel('steps')
mypl.ylabel('reward')
for perf_file in perf_files:
    with open(perf_file[0], 'r') as f:
        perf = json.load(f)
        print('loaded '+perf_file[0])
    if 'env_ok_eval' in perf:
        valid_eval_episodes_ratio=round(sum(perf['env_ok_eval'])/len(perf['env_ok_eval']),2)
        print('valid_eval_episodes_ratio = '+ str(valid_eval_episodes_ratio))
    rewards_smooth=rolling_average(perf['eval_rewards'],n=wind)
    rewards_smooth=np.concatenate([np.array((wind-1)*[rewards_smooth[0]]),rewards_smooth])
    #mypl.title('Pong_advice')
    if dbg_flg:
        print('length of rewards_smooth: '+str(rewards_smooth.shape[0]))
    mypl.plot(perf['eval_steps'],rewards_smooth,label=perf_file[1])
    mypl.legend()
mypl.grid(True)
print('generating plots ...')
mypl.savefig('perfplot_1.png')
a=7