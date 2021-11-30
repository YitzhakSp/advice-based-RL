import pandas as pd

models_dir='../models/gridworld'
q_tab=pd.read_pickle(models_dir+'/qfunc.pkl')
tmp=6