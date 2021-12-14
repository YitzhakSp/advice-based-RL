models_dir='models/Gridworld'
eps_start=1.0
eps_final=0.01
eps_decay_duration=1000
gamma=0.9
num_episodes=1001
max_episode_steps=100
eval_freq=10
save_freq=1000
q_init=0
use_advice=True
qfunc_adv_file=models_dir+'/qfunc_opt_arch_3.pkl'
advice_criterion='crit'
imp_trh=1e-2
crit_trh=0.5
limited_advice=True
advice_budget=1e2
save_model=True
verbose=False
dbg=True