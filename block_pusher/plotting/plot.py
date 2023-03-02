import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

if len(sys.argv) < 3:
    assert False, "usage: python plot.py [logdir] [key]"
directory = sys.argv[1]
KEY = sys.argv[2] # e.g. 'cumulative_reward'
files = sorted(os.listdir(directory), key=lambda x:x[::-1])
EP_LEN = 1000
NUM_ROBOTS = 100
filedatas = [pickle.load(open(directory+'/'+f+'/run_stats.pkl','rb'))[KEY] for f in files]
minlen = min([len(fd) for fd in filedatas])
filedatas = [fd[:minlen] for fd in filedatas]
if len(sys.argv) == 4:
    KEY2 = sys.argv[3]
    filedatas2 = [pickle.load(open(directory+'/'+f+'/run_stats.pkl', 'rb'))[KEY2] for f in files]
    filedatas2 = [fd[:minlen] for fd in filedatas2]
#plt.style.use('seaborn-darkgrid')
# load data
colors = ['green', 'gold', 'blue', 'orange', 'purple', 'pink', 'brown']
for i in range(0,len(files),3):
    label = '{}'.format(files[i][files[i].rindex('_')+1:])
    data = np.array(filedatas[i:i+3])
    if len(sys.argv) == 4:
        data2 = np.array(filedatas2[i:i+3])
        data = 100*(data/(data2+20))
    if KEY == 'cumulative_reward':
        plt.plot([v*EP_LEN/(i+1)/NUM_ROBOTS for i,v in enumerate(data.mean(axis=0))], label=label, color=colors[i//3])
        plt.fill_between(np.arange(minlen), [v*EP_LEN/(i+1)/NUM_ROBOTS for i,v in enumerate(data.mean(axis=0)-data.std(axis=0))], [v*EP_LEN/(i+1)/NUM_ROBOTS for i,v in enumerate(data.mean(axis=0)+data.std(axis=0))], alpha=0.2, color=colors[i//3])
    else:
        plt.plot(data.mean(axis=0), label=label, color=colors[i//3])
        plt.fill_between(np.arange(minlen), data.mean(axis=0)-data.std(axis=0), data.mean(axis=0)+data.std(axis=0), alpha=0.2, color=colors[i//3])
#plt.xlabel('Timestep')
#plt.legend()
if len(sys.argv) == 4:
    #plt.title('{}/{}'.format(KEY, KEY2))
    #plt.ylim(0,1)
    plt.savefig('ratio.jpg'.format(KEY, KEY2), bbox_inches='tight')
else:
    #plt.title('{}'.format(KEY))
    plt.savefig('{}.jpg'.format(KEY), bbox_inches='tight')
        
