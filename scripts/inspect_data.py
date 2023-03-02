import pickle
import cv2
import sys
import numpy as np

p = pickle.load(open(sys.argv[1], 'rb'))
obs = p['obs']
act = p['act']
rew = p['rew']
done = p['done']
obs2 = p['obs2']
obsnew, actnew, obs2new, rewnew, donenew = list(), list(), list(), list(), list()
print(len(obs))

for i in range(len(obs)):
    print("timestep", i, "act", act[i])
    #print("timestep", i, "dx", -act[i][1], "dy", act[i][0]) # convert to standard axes
    cv2.imshow('', obs[i])
    key = cv2.waitKey()
    if key == ord("r"): # reject sample
        continue
    else:
        obsnew.append(obs[i])
        actnew.append(act[i])
        obs2new.append(obs2[i])
        rewnew.append(rew[i])
        donenew.append(done[i])
pickle.dump({'obs': np.stack(obsnew), 'act': np.stack(actnew), 'obs2': np.stack(obs2new), 
   'rew': np.array(rewnew), 'done': np.array(donenew)}, open('clean_demos.pkl', 'wb'))
