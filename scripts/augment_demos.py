"""
This script extracts task demos from a fully supervised run (num robots=1, num humans=1, free_humans=False)
"""
import pickle
import sys
import numpy as np
import isaacgym
from parallel_robotics.agents.impl.utils import augment

AUG_FACTOR = 10

p = pickle.load(open(sys.argv[1], 'rb')) # data
obsnew, actnew, rewnew, donenew, obs2new = list(), list(), list(), list(), list()
for i in range(len(p['obs'])):
    img = p['obs'][i]
    img2 = p['obs2'][i]
    imgstack = np.hstack(((img*255).astype(np.uint8), (img2*255).astype(np.uint8)))
    for _ in range(AUG_FACTOR):
        augmented = augment(imgstack)
        img_mod = (augmented[:,:img.shape[1],:] / 255.).astype(np.float32)
        img2_mod = (augmented[:,img.shape[1]:,:] / 255.).astype(np.float32)
        obsnew.append(img_mod)
        actnew.append(p['act'][i])
        rewnew.append(p['rew'][i])
        donenew.append(p['done'][i])
        obs2new.append(img2_mod)
pickle.dump({'obs': np.stack(obsnew), 'act': np.stack(actnew), 'obs2': np.stack(obs2new), 
    'rew': np.array(rewnew), 'done': np.array(donenew)}, open('aug_demos.pkl', 'wb'))
