import os
base_datadir = 'dataset/'
model_savedir = 'models/'
results_savedir = 'results/'

if not os.path.exists(base_datadir):
    os.makedirs(base_datadir)
if not os.path.exists(model_savedir):
    os.makedirs(model_savedir)
if not os.path.exists(results_savedir):
    os.makedirs(results_savedir)
