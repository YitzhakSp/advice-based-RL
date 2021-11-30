from utils.other_utils import *
import json
import matplotlib.pyplot as plt

#thismodel_dir='simulations/advice/soft_treshold'
perf_file='../models/gridworld/perf_1.json'
x_axis='episodes'
max_steps_flg=False
max_steps=2e6
wind=1
plot_flg=True
steps_pt_sum=0
with open(perf_file, 'r') as f:
    perf = json.load(f)
valid_eval_episodes_ratio=round(sum(perf['env_ok_eval'])/len(perf['env_ok_eval']),2)
print('valid_eval_episodes_ratio = '+ str(valid_eval_episodes_ratio))
rewards_smooth_train=rolling_average(perf['train_scores'],n=wind)
rewards_smooth_train=np.concatenate([np.array((wind-1)*[rewards_smooth_train[0]]),rewards_smooth_train])
rewards_smooth_eval=rolling_average(perf['eval_scores'],n=wind)
rewards_smooth_eval=np.concatenate([np.array((wind-1)*[rewards_smooth_eval[0]]),rewards_smooth_eval])
mypl=plt
if max_steps_flg:
    mypl.xlim(right=max_steps)
mypl.ylabel('reward')
#mypl.title('Pong_advice')
assert (x_axis=='steps' or x_axis=='episodes')
if x_axis=='steps':
    mypl.xlabel('steps')
    mypl.plot(perf['eval_steps'],rewards_smooth)
elif x_axis=='episodes':
    mypl.xlabel('episodes')
    mypl.plot(perf['eval_episodes'],rewards_smooth_eval,label='eval')
    mypl.plot(rewards_smooth_train,label='train')
plt.legend()


'''
ind=firstind_above(rewards_smooth,9.99)
steps_pt=perf['eval_steps'][ind] # steps until performance treshold
steps_pt_sum+=steps_pt
print('steps_pt_{}: {}'.format(seed,steps_pt))
print('generating plot ...')
mypl.axvline(x= steps_pt, color = 'r')
print('steps_pt: ',steps_pt )
'''
mypl.grid(True)
# attention: always remember this:
#mypl.show()
print('saving plot ...')
mypl.savefig('perfplot_1.png')
a=7