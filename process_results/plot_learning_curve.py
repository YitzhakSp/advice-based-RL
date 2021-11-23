from utils.other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/advice/soft_treshold'
perf_file='../simulations/Gopher/advice/7e-2/perf_1.json'
max_steps_flg=False
max_steps=2e6
wind=100
plot_flg=True
steps_pt_sum=0
with open(perf_file, 'r') as f:
    perf = json.load(f)
valid_eval_episodes_ratio=round(sum(perf['env_ok_eval'])/len(perf['env_ok_eval']),2)
print('valid_eval_episodes_ratio = '+ str(valid_eval_episodes_ratio))
rewards_smooth=rolling_average(perf['eval_rewards'],n=wind)
rewards_smooth=np.concatenate([np.array((wind-1)*[rewards_smooth[0]]),rewards_smooth])
mypl=plt
if max_steps_flg:
    mypl.xlim(right=max_steps)
mypl.xlabel('steps')
mypl.ylabel('reward')
#mypl.title('Pong_advice')
mypl.plot(perf['eval_steps'],rewards_smooth)
ind=firstind_above(rewards_smooth,9.99)
steps_pt=perf['eval_steps'][ind] # steps until performance treshold
steps_pt_sum+=steps_pt
print('steps_pt_{}: {}'.format(seed,steps_pt))
print('generating plot ...')
mypl.axvline(x= steps_pt, color = 'r')
mypl.grid(True)
# attention: always remember this:
#mypl.show()
mypl.savefig('perfplot_1.png')
print('steps_pt: ',steps_pt )
a=7