import os
import pandas as pd
import numpy as np
import dill
from random import sample
from keras.layers import Dense, Lambda ,Input, LSTM, GRU, Dropout, Conv1D, Bidirectional, Flatten, Reshape, Permute, concatenate
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

def columnar_R2(mat_predict,mat_true):
    top=mat_predict.shape[1]
    out=np.zeros(top)
    for i in range(top):
        out[i]=1-np.linalg.norm(mat_predict[:,i]-mat_true[:,i])/np.linalg.norm(mat_predict[:,i]-mat_true[:,i].mean())
    return out


names=os.listdir('Train Data')

score_log={}

for name in names:
	name=name.split('.')
	if name[-1]!="csv":
		continue
	name=name[0]
	df=pd.read_csv('Train Data/'+name+'.csv').drop(columns=['time'])

	assert df.isna().sum().sum()==0, name+" has NANs"

	data_info={}
	data_info['primary']=name
	data_info['num_features']=df.shape[1]
	data_info['train_time']=12
	data_info['predict_time']=4
	data_info['num_samples']=df.shape[0]

	X=np.zeros((data_info['num_samples'], data_info['train_time'], df.shape[1]))
	Y=np.zeros((data_info['num_samples'], data_info['predict_time']))
	for i in range(data_info['num_samples']-(data_info['train_time']+data_info['predict_time']+1)):
	    X[i,:,:]=df.iloc[i:i+data_info['train_time']].values
	    Y[i,:]=np.asarray(df[name+'_scaled_demand'][i+data_info['train_time']:i+data_info['train_time']+data_info['predict_time']].values)

	idx=sample(range(len(X)),9000)
	train_idx=idx[:7000]
	valid_idx=idx[7000:]

	data_info['num_train_samples']=len(train_idx)
	data_info['num_valid_samples']=len(valid_idx)
	data_info['train_idx']=train_idx
	data_info['valid_idx']=valid_idx

	X_train=np.zeros(
	    (data_info['num_train_samples'],
	     data_info['train_time'],
	     data_info['num_features']))
	Y_train=np.zeros(
	    (data_info['num_train_samples'],
	     data_info['predict_time']))

	X_valid=np.zeros(
	    (data_info['num_valid_samples'],
	     data_info['train_time'],
	     data_info['num_features']))
	Y_valid=np.zeros(
	    (data_info['num_valid_samples'],
	     data_info['predict_time']))

	for i,ind in enumerate(data_info['train_idx']):
	    X_train[i,:,:]=X[ind,:,:]
	    Y_train[i,:]=Y[ind,:]
	for i,ind in enumerate(data_info['valid_idx']):
	    X_valid[i,:,:]=X[ind,:,:]
	    Y_valid[i,:]=Y[ind,:]


	inp= Input(shape = (data_info['train_time'],data_info['num_features'],))

	trans=Permute((2,1))(inp)

	GRU1 = GRU(20, return_sequences = True)(trans)
	drop1= Dropout(rate=.3)(GRU1)
	conv1= Conv1D(20,3,padding="valid")(drop1)
	LSTM1= LSTM(20, return_sequences=True)(conv1)
	flat1=Flatten()(LSTM1)

	dense1=Dense(20,activation="linear")(flat1)

	out = Dense(data_info['predict_time'],activation="linear")(dense1)

	earlystop = EarlyStopping(monitor='val_loss', min_delta=.01, patience=2, verbose=1, mode='auto')

	model2 = Model(inputs = inp, outputs = out)
	model2.compile(loss = "mean_squared_error", optimizer = Adam(lr = .001, decay = 0))

	history = model2.fit(X_train, 
                    Y_train, 
                    batch_size = 16, 
                    epochs = 20, 
                    verbose=1,
                    validation_data=(X_valid,Y_valid), 
                    callbacks=[earlystop])

	Y_predict=model2.predict(X_valid)

	score_log[name]=columnar_R2(Y_predict,Y_valid)

	dill.dump(score_log,open('model_R2_scores.pkd','wb'))


