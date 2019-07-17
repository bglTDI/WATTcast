import numpy as np
import dill
from sklearn.model_selection import train_test_split
import time

from keras.layers import Dense, Lambda ,Input, LSTM, GRU, Dropout, Conv1D, Bidirectional, Flatten, Reshape, Permute, concatenate
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.hyperparameters import UniformFloatHyperparameter as UFH
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as UIH

class WTTWorker(Worker):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self.batch_size=16

		self.data_info=dill.load(open('Data/data_info.pkl','rb'))

		self.X_train=dill.load(open('Data/X_train.pkl','rb'))
		self.X_valid=dill.load(open('Data/X_valid.pkl','rb'))
		self.Y_train=dill.load(open('Data/Y_train.pkl','rb'))
		self.Y_valid=dill.load(open('Data/Y_valid.pkl','rb'))
	  
	def compute(self,config,budget,working_directory,*args,**kwargs):

		####################################################################
		# Input & Split #####################################################
		#####################################################################
		concatenated_input= Input(shape = (self.data_info['train_time'],self.data_info['num_features'],))

		#####################################################################
		# Weather Feature Prong #############################################
		#####################################################################
		
		transpose_input=Permute((2,1))(concatenated_input)

		temporal_GRU1 = GRU(	config['num_temporal_GRU1'],
										recurrent_dropout=config['GRU_dropout_rate'], 
										return_sequences = True)(transpose_input)
		temporal_dropout_1 = Dropout(rate=config['dropout_layer_rate'])(temporal_GRU1)
		temporal_conv1 = Conv1D(	config['num_temporal_conv'],
											config['size_temporal_conv'],
											padding="valid")(temporal_dropout_1)
		temporal_GRU2 = GRU(	config['num_temporal_GRU2'], 
										recurrent_dropout=config['GRU_dropout_rate'],
									return_sequences=True)(temporal_conv1)
		temporal_flatten =Flatten()(temporal_GRU2)		
		
		#####################################################################
		# Decision Layers ###################################################
		#####################################################################

		dense1=Dense(	config['num_dense1'],
						activation="linear")(temporal_flatten)
		final_dropout=Dropout(	rate=config['dropout_layer_rate'])(dense1)
		dense2=Dense(	config['num_dense2'],
						activation='linear')(final_dropout)

		output = Dense(	self.data_info['predict_time'], 
						activation="linear")(dense2)
				
		#####################################################################
		# Model Definition ##################################################
		#####################################################################

		model=Model(concatenated_input,output)
		
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=config['learning_rate']))

		model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, epochs=int(budget),verbose=0)

		train_score=model.evaluate(self.X_train,self.Y_train)
		valid_score=model.evaluate(self.X_valid,self.Y_valid)
		Y_predict=model.predict(self.X_valid)

		def columnar_R2(mat_predict,mat_true):
			out=[0,0,0,0]
			for i in range(4):
				out[i]=1-np.linalg.norm(mat_predict[:,i]-mat_true[:,i])/np.linalg.norm(mat_predict[:,i]-mat_true[:,i].mean())
			return out

		weighted_R2=-sum(columnar_R2(Y_predict,self.Y_valid))/4

		return({'loss': weighted_R2, 
				'info': {'num_pars':model.count_params(), 'train':train_score, 'valid':valid_score}})


	@staticmethod
	def get_configspace():
		cs=CS.ConfigurationSpace()
		learning_rate=UFH(				'learning_rate',
										lower=1e-6,upper=1e-1,
										default_value=1e-2,log=True)

		GRU_dropout_rate=UFH(			'GRU_dropout_rate',
										lower=.1,upper=.4,
										default_value=.3,log=True)

		dropout_layer_rate=UFH(			'dropout_layer_rate',
										lower=.1,upper=.4,
										default_value=.3,log=True)

		num_temporal_GRU1=UIH(			'num_temporal_GRU1',
										lower=10,upper=50,
										default_value=20,log=False)

		num_temporal_conv=UIH(			'num_temporal_conv',
										lower=10,upper=50,
										default_value=15,log=False)

		size_temporal_conv=UIH(			'size_temporal_conv',
										lower=2,upper=7,
										default_value=3,log=False)

		num_temporal_GRU2=UIH(			'num_temporal_GRU2',
										lower=10,upper=50,
										default_value=20,log=False)

		num_dense1=UIH(					'num_dense1',
										lower=10,upper=50,
										default_value=20,log=False)

		num_dense2=UIH(					'num_dense2',
										lower=10,upper=50,
										default_value=20,log=False)

		cs.add_hyperparameters([		learning_rate,
										GRU_dropout_rate,
										dropout_layer_rate,
										num_temporal_GRU1,
										num_temporal_conv,
										size_temporal_conv,
										num_temporal_GRU2,
										num_dense1,
										num_dense2])
		return cs