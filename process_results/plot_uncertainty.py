from utils.other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/advice/soft_treshold'
thismodel_dir='../simulations/Gopher/no_advice'
print('loading performance data from '+ thismodel_dir)
max_steps_flg=False
max_steps=2e6
wind=100
plot_flg=True
steps_pt_sum=0
with open(thismodel_dir+'/perf_1_withuncert.json', 'r') as f:
    perf = json.load(f)
valid_eval_episodes_ratio=round(sum(perf['env_ok_eval'])/len(perf['env_ok_eval']),2)
print('valid_eval_episodes_ratio = '+ str(valid_eval_episodes_ratio))
rewards_smooth=rolling_average(perf['eval_rewards'],n=wind)
rewards_smooth=np.concatenate([np.array((wind-1)*[rewards_smooth[0]]),rewards_smooth])
mypl=plt
mypl.xlabel('steps')
mypl.ylabel('uncert')
#mypl.title('Pong_advice')
mypl.subplot(2, 2, 1)
mypl.plot(perf['min_uncertainty'],label='min uncert')
mypl.legend()
mypl.subplot(2, 2, 2)
mypl.plot(perf['max_uncertainty'],label='max uncert')
mypl.legend()
mypl.subplot(2, 2, 3)
mypl.plot(perf['avg_uncertainty'],label='avg uncert')
mypl.legend()
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('uncertplot_1.png')

a=7