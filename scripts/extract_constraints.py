"""
This script extracts constraint demos from a constraint generation run
"""
import pickle
import sys
import numpy as np

p = pickle.load(open(sys.argv[1], 'rb')) # raw data log file
obs, act, rew, done, next_obs = list(), list(), list(), list(), list()
for j in range(len(p[0]['state'])):
    for i in range(len(p)):
        if p[i]['real_act'][j] is None:
            continue
        obs.append(p[i]['state'][j])
        act.append(p[i]['real_act'][j])
        rew.append(p[i]['info'][j]['constraint'])
        done.append(p[i]['done'][j])
        if i < len(p) - 1:
            next_obs.append(p[i+1]['state'][j])
    next_obs.append(np.zeros(obs[0].shape))
pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'obs2': np.stack(next_obs), 
    'rew': np.array(rew), 'done': np.array(done)}, open('constraints.pkl', 'wb'))
