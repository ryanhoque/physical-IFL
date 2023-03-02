import pickle
import sys
import numpy as np

p1 = pickle.load(open(sys.argv[1], 'rb'))
p2 = pickle.load(open(sys.argv[2], 'rb'))

pickle.dump({'obs': np.vstack((p1['obs'], p2['obs'])), 'act': np.concatenate((p1['act'], p2['act'])), 'obs2': np.vstack((p1['obs2'], p2['obs2'])), 
   'rew': np.concatenate((p1['rew'], p2['rew'])), 'done': np.concatenate((p1['done'], p2['done']))}, open('output.pkl', 'wb'))
