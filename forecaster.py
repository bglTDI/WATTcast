import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="Progress:")
import random as rand

from keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Bidirectional, Flatten, Reshape, Permute, concatenate
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from hpbandster.core.worker import Worker

import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

dirs=[name for name in os.listdir(".") if os.path.isdir(name)]
try:
	last_run_num=sorted([int(guy.split("_")[1]) for guy in dirs if guy.split("_")[0]=="Run"])[-1]
except:
	last_run_num=-1

current_run=last_run_num+1
run_dir="./Run_"+str(current_run)

os.mkdir(run_dir)



warm_start=False
make_samples=True

if make_samples:
	df=pd.read_csv('./scrubbed_dataframe.csv').drop("Unnamed: 0",axis=1)

	train_length=50000
	test_length=10000
	train_time=12
	prediction_time=8
	per=train_time+prediction_time
	breadth=df.shape[1]

	demand_tensor=np.zeros((train_length,breadth,per))


	for k in tqdm(range(train_length)):
		rand_ind=rand.choice(range(df.shape[0]-200))
		demand_tensor[k,:,:]=df[rand_ind:rand_ind+per].T.to_numpy()

	np.save(run_dir+'/dump1.npy',np.concatenate([demand_tensor[i,:,:] for i in range(train_length)],axis=1))

	valid_demand_tensor=np.zeros((test_length,breadth,per))

	for k in tqdm(range(test_length)):
		rand_ind=rand.choice(range(df.shape[0]-200))
		valid_demand_tensor[k,:,:]=df[rand_ind:rand_ind+per].T.to_numpy()

	np.save(run_dir+'/dump2.npy',np.concatenate([valid_demand_tensor[i,:,:] for i in range(test_length)],axis=1))

def weighted_RMS(layer,discount,prediction_time):
	def loss(y_true,y_pred):
		time_mat=K.variable(np.diagflat([discount**i for i in range(prediction_time)]))
		return K.mean(K.square(K.dot(y_true-y_pred,time_mat)))
	return loss

def load_and_tensorify(filename,length,per):
	huge_mat=np.load(filename)
	out=np.zeros((length,40,per))
	for i in range(length):
		out[i,:,:]=huge_mat[:,i*per:(i+1)*per]
	return out

class KerasWorker(Worker):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		self.batch_size = 128
		self.epochs = 20
		self.validation_split = 0.2

		self.train_length=train_length
		self.test_length=test_length

		self.train_time=train_time
		self.prediction_time=prediction_time
		self.per=self.train_time+self.prediction_time
		self.breadth=40

		train_data=load_and_tensorify(run_dir+'/dump1.npy',self.train_length,self.per)
		test_data=load_and_tensorify(run_dir+'/dump2.npy',self.test_length,self.per)

		
		self.x_train, self.y_train = train_data[:,:,:self.train_time], train_data[:,:,self.train_time:]
		self.x_test, self.y_test = test_data[:,:,:self.train_time], test_data[:,:,self.train_time:]

		self.early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2)

		self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
												patience=2, 
												verbose=0, 
												factor=0.5, 
												min_lr=1e-8)
	
	def compute(self, config, budget, working_directory, *args, **kwargs):

		inp= Input(shape = (self.breadth,self.train_time,))

		trans=Permute((2,1))(inp)

		LSTM1 = LSTM(config['units1'], return_sequences = True)(inp)
		drop1= Dropout(rate=config['dr'])(LSTM1)
		conv1= Conv1D(config["conv_units1"],config['kernel_size1'],padding="valid")(drop1)
		flat1=Flatten()(conv1)

		LSTM2 = LSTM(config['units2'], return_sequences = True)(trans)
		drop2= Dropout(rate=config['dr'])(LSTM2)
		conv2= Conv1D(config["conv_units2"],config['kernel_size2'],padding="valid")(drop2)
		flat2=Flatten()(conv2)

		concat=concatenate([flat1,flat2])

		dense1 = Dense(self.prediction_time*self.breadth,activation="sigmoid")(concat)
		out=Reshape((self.breadth,self.prediction_time))(dense1)

		model = Model(inputs = inp, outputs = out)
		model.compile(loss = "mean_squared_error", optimizer = Adam(lr = config['lr'], decay = config['lr_dr']))
		# model.summary()
		history = model.fit(self.x_train, self.y_train, batch_size = self.batch_size, epochs = int(budget), validation_split=self.validation_split,verbose=2,callbacks=[self.early_stop])

		train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
		val_score = model.evaluate(self.x_test, self.y_test, verbose=0)
				
		return ({'loss': val_score, 'info': {'number of parameters': model.count_params(), "validation score":val_score }})
					
	@staticmethod
	def get_configspace():
		cs = CS.ConfigurationSpace()
		lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value=1e-2, log=True)
		lr_dr = CSH.UniformFloatHyperparameter('lr_dr', lower=1e-6, upper=1e-1, default_value=1e-2, log=True)
		dr = CSH.UniformFloatHyperparameter('dr', lower=.2, upper=.8, default_value=.5, log=False)
		units1 = CSH.UniformIntegerHyperparameter('units1', lower=4, upper=128, default_value=32, log=False)
		units2 = CSH.UniformIntegerHyperparameter('units2', lower=4, upper=128, default_value=32, log=False)
		conv_units1 = CSH.UniformIntegerHyperparameter('conv_units1', lower=4, upper=64, default_value=5, log=False)
		conv_units2 = CSH.UniformIntegerHyperparameter('conv_units2', lower=4, upper=64, default_value=5, log=False)
		kernel_size1 = CSH.UniformIntegerHyperparameter('kernel_size1', lower=1, upper=10, default_value=5, log=False)
		kernel_size2 = CSH.UniformIntegerHyperparameter('kernel_size2', lower=1, upper=10, default_value=5, log=False)
		# discount=CSH.UniformFloatHyperparameter('discount', lower=.6, upper=.9, default_value=.8, log=False)

		cs.add_hyperparameters([lr, lr_dr, dr, units1, conv_units1, kernel_size1, units2, conv_units2, kernel_size2]) 
						
		return cs


# min_budget = float(1)
# max_budget = float(5)
# n_iterations = int(48)
# # worker = 'store_true'
# run_id = 'second'
# # nic_name = 'lo'
# shared_directory = '.'

# # host = hpns.nic_name_to_host(nic_name)


# NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None, working_directory='.')
# NS.start()

# w = KerasWorker(run_id='first', nameserver='127.0.0.1')
# w.run(background=True)

# bohb = BOHB(configspace = w.get_configspace(), run_id = 'first',
# 			nameserver='127.0.0.1',
# 			min_budget=min_budget, max_budget=max_budget,eta=4)
# res = bohb.run(n_iterations=n_iterations)

# with open(os.path.join('.', 'results.pkl'), 'wb') as fh:
# 	pickle.dump(res, fh)

# bohb.shutdown(shutdown_workers=True)
# NS.shutdown()

# id2config = res.get_id2config_mapping()
# incumbent = res.get_incumbent_id()

# print('Best found configuration:', id2config[incumbent]['config'])
# print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
# print('A total of %i runs where executed.' % len(res.get_all_runs()))
# print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/max_budget))


# NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None, working_directory='.')
# NS.start()

# workers = []
# for i in range(4):
#     w = KerasWorker(nameserver='127.0.0.1',run_id='example1', id=i)
#     w.run(background = 4)
#     workers.append(w)

# # w = KerasWorker(run_id='first', nameserver='127.0.0.1')
# # w.run(background=True)

# bohb = BOHB(configspace = w.get_configspace(), run_id = 'example1',
#             nameserver='127.0.0.1',
#             min_budget=min_budget, max_budget=max_budget,)
# res = bohb.run(n_iterations=n_iterations, min_n_workers = 4)

# with open(os.path.join('.', 'results.pkl'), 'wb') as fh:
#     pickle.dump(res, fh)

# bohb.shutdown(shutdown_workers=True)
# NS.shutdown()

# id2config = res.get_id2config_mapping()
# incumbent = res.get_incumbent_id()

# all_runs = res.get_all_runs()

# print('Best found configuration:', id2config[incumbent]['config'])
# print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
# print('A total of %i runs where executed.' % len(res.get_all_runs()))
# print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
# print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in all_runs])/max_budget))
# print('The run took  %.1f seconds to complete.'%(all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))


parser = argparse.ArgumentParser(description='Two pronged LSTM model')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=1)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=4)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default=run_dir)
parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.', default='./Run_'+str(last_run_num))


args=parser.parse_args()

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)

if args.worker:
	import time
	time.sleep(5)	# short artificial delay to make sure the nameserver is already running
	w = KerasWorker(run_id=args.run_id, host=host, timeout=120)
	w.load_nameserver_credentials(working_directory=args.shared_directory)
	w.run(background=False)
	exit(0)

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)


# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = KerasWorker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)


# Let us load the old run now to use its results to warmstart a new run with slightly
# different budgets in terms of datapoints and epochs.
# Note that the search space has to be identical though!

if warm_start:
	previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)
	bohb = BOHB(  configspace = w.get_configspace(),
			  run_id = args.run_id,
			  host=host,
			  nameserver=ns_host,
			  nameserver_port=ns_port,
			  result_logger=result_logger,
			  min_budget=args.min_budget, max_budget=args.max_budget,	
  			  previous_result = previous_run,			
		   )
else:
		# Run an optimizer
	bohb = BOHB(  configspace = w.get_configspace(),
				  run_id = args.run_id,
				  host=host,
				  nameserver=ns_host,
				  nameserver_port=ns_port,
				  result_logger=result_logger,
				  min_budget=args.min_budget, max_budget=args.max_budget,			
			   )

res = bohb.run(n_iterations=args.n_iterations)

# store results
with open(os.path.join(args.shared_directory, '2P_LSTM_results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()