import pandas as pd
import numpy as np
import dill

import os
import argparse
import logging

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB



#Toggle the following to un/mute the nameserver chatter
# logging.basicConfig(level=logging.WARNING)

#Warm start option
# warm_start=True

# #Check for previous runs, set pointer to latest, make new results directory
# run_cnt=-1
# file_list=os.listdir()
# for file in file_list:
# 	temp=set(file.split('_'))
# 	if {'WTTcast','run'}.issubset(temp):
# 		run_cnt+=1

# os.mkdir('WTTcast_run_'+str(run_cnt+1))

# Data preprocessing
make_slices=True

if make_slices:
	df=pd.read_csv('Data/BPAT_trial.csv').drop(columns=['Unnamed: 0','time']).interpolate()
	X=np.zeros((df.shape[0]-20,12,df.shape[1]))
	Y=np.zeros((df.shape[0]-20,4))
	for i in range(df.shape[0]-20):
		X[i,:,:]=df.iloc[i:i+12].values
		Y[i,:]=np.asarray(df['BPAT'][i+12:i+16].values) 


	dill.dump(X,open("Data/BPAT_X_slices.pkl",'wb'))
	dill.dump(Y,open("Data/BPAT_Y_slices.pkl",'wb'))

test_train_richness=.8
top=int(test_train_richness*len(X))

idx=list(range(len(X)))
np.random.shuffle(idx)
train_idx=idx[:top]
valid_idx=idx[top:]	

data_info={}
data_info['columns']=list(df.columns)
data_info['primary']='BPAT'
data_info['num_features']=13
data_info['num_stations']=7
data_info['train_time']=12
data_info['predict_time']=4
data_info['num_train_samples']=len(train_idx)
data_info['num_valid_samples']=len(valid_idx)
data_info['train_idx']=train_idx
data_info['valid_idx']=valid_idx

dill.dump(data_info,open("Data/data_info.pkl",'wb'))

# Import a worker class
from WTTWorker import WTTWorker as worker

#Build an argument parser 
warm_start_check=False       
parser = argparse.ArgumentParser(description='WTTcast - sequential execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=10)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=30)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')
# if warm_start and (run_cnt>0):
# 	parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.', default='WTTcast_run_'+str(run_cnt))
# 	warm_start_check=True

args=parser.parse_args()

#Define a realtime result logger
result_logger = hpres.json_result_logger(directory='.', overwrite=False)


#Start a nameserver
NS = hpns.NameServer(run_id='WTTcast', host='127.0.0.1', port=None)
NS.start()

#Start the workers
workers=[]
for i in range(args.n_workers):
    w = worker(nameserver='127.0.0.1',run_id='WTTcast', id=i)
    w.run(background=True)
    workers.append(w)

warm_start_check=False

#Define and run an optimizer
if warm_start_check:
	bohb = BOHB(  configspace = w.get_configspace(),
	              run_id = 'tagster',
	              result_logger=result_logger,
	              min_budget=args.min_budget, max_budget=args.max_budget,
	              previous_result = previous_run
	           )
else:
	bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'WTTcast',
              result_logger=result_logger,
              min_budget=args.min_budget, max_budget=args.max_budget) 

res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

#Shutdown the nameserver
bohb.shutdown(shutdown_workers=True)
NS.shutdown()