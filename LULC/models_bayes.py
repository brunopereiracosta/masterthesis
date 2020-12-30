import tensorflow as tf
import numpy as np
from data_pipeline import Extract
from arguments import *

#models
class MCDropout(tf.keras.layers.Dropout):
	def call(self,inputs):
		return super().call(inputs,training=True)
class MCalphaDropout(tf.keras.layers.AlphaDropout):
	def call(self,inputs):
		return super().call(inputs,training=True)		
def last_time_step_mse(y_true,y_pred):
	ltsm = tf.keras.metrics.mean_squared_error(y_true[:,-1],y_pred[:,-1])
	return ltsm
# he_avg_init=tf.keras.initializers.VarianceScaling(scale=2.,mode='fan_avg',distribution='uniform')
def baseline_model(x_train, y_train, x_val, y_val, data, epochs, output_neurons=output_neurons):
	# data argument should be eliminated and input_shape would be = [window_size,x_train.shape[1]]
	window_size = 7

	print('####### Linear Regression MSE #######')
	# x_train, y_train, x_val, y_val, x_test, y_test, scaler2 = split_norm_batch(data=data,fold_selector=fold_selector)
	y_train_extracted = np.array(Extract(y_train))
	y_val_extracted = np.array(Extract(y_val))
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape=[window_size,data.shape[1]]))
	model.add(tf.keras.layers.Dense(output_neurons))
	model.compile(optimizer='adam',loss='mse')
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
												patience=12, mode='min')
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
										factor=0.5, patience=6, min_lr=0.001,mode='min')
	
	hist = model.fit(x_train, y_train_extracted, epochs=epochs,
		validation_data=(x_val, y_val_extracted), callbacks = [early_stop,reduce_lr],verbose=0)
	mse_baseline = {'MSE_baseline' : round(model.evaluate(x_val, y_val_extracted),3)}
	return mse_baseline

def build_model_Simple(hp, n_feature=n_feature, output_neurons=output_neurons,timesteps = None, stateful = False):
	
	kernel_initializer=hp.Choice('kernel_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
	if kernel_initializer in ['glorot_uniform', 'glorot_normal']:
		activation=hp.Choice('activation',['tanh', 'sigmoid'],
		parent_name='kernel_initializer',
		parent_values=['glorot_uniform', 'glorot_normal'])
	elif kernel_initializer in ['he_uniform', 'he_normal']:
		activation=hp.Choice('activation',['elu','relu'],
		parent_name='kernel_initializer',
		parent_values=['he_uniform', 'he_normal'])
	elif kernel_initializer in ['lecun_uniform','lecun_normal']:
		activation=hp.Choice('activation',['selu'],
		parent_name='kernel_initializer',
		parent_values=['lecun_uniform','lecun_normal'])

	# if activation == 'leakyrelu':
		# activation = tf.keras.layers.LeakyReLu(alpha=0.2)

	# kernel_regularizer=tf.keras.regularizers.l2(0.01)
	kernel_constraint=tf.keras.constraints.MaxNorm(1.)
	# rate=hp.Choice('rate',[0., 0.1, 0.2, 0.4, 0.6])
	n_neurons=hp.Int('n_neurons',min_value=1,max_value=100,step=5)
	optimizer=hp.Choice('optimizer',['opt1','opt2','opt3','opt4','opt5'])
	# learning_rate=hp.Choice('learning_rate',[0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
	learning_rate=0.2 # decreasing during stable periods, see callback
	loss=hp.Choice('loss',['mse', 'mae'])
	# recurrent_dropout=hp.Choice('recurrent_dropout',[0., 0.1, 0.2, 0.4, 0.6])

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.SimpleRNN(n_neurons,
											kernel_initializer=kernel_initializer,
											activation=activation,
											kernel_constraint=kernel_constraint,
											# kernel_regularizer=kernel_regularizer,
											return_sequences=True,
											stateful=stateful,
											input_shape=(timesteps,n_feature)))
	# model.add(tf.keras.layers.Dropout(rate=rate)) 
	for i in range(hp.Int('num_layers', 0, 3)):
		model.add(tf.keras.layers.SimpleRNN(units=hp.Int('n_neurons' + str(i),
											min_value=1,
											max_value=100,
											step=5),
											# recurrent_dropout=recurrent_dropout,
											kernel_constraint=kernel_constraint,
											# kernel_regularizer=kernel_regularizer,
											kernel_initializer=kernel_initializer,
											activation=activation,
											return_sequences=True,
											stateful=stateful))
		# model.add(tf.keras.layers.Dropout(rate=hp.Choice('rate' + str(i+1),[0., 0.1, 0.2, 0.4, 0.6])))	
	model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_neurons)))

	if optimizer == 'opt1':
		optimizer = tf.keras.optimizers.RMSprop(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt2':
		optimizer = tf.keras.optimizers.SGD(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate,nesterov=hp.Choice('nesterov',['True','False']))
	elif optimizer == 'opt3':
		optimizer = tf.keras.optimizers.Adam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt4':
		optimizer = tf.keras.optimizers.Adamax(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt5':
		optimizer = tf.keras.optimizers.Nadam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	
	model.compile(optimizer=optimizer,loss=loss,sample_weight_mode='temporal',metrics=[last_time_step_mse])
	
	print(model.summary())
	return model
def build_model_LSTM(hp, n_feature=n_feature, output_neurons=output_neurons, timesteps = None, stateful = False):
	
	kernel_initializer=hp.Choice('kernel_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
	if kernel_initializer in ['glorot_uniform', 'glorot_normal']:
		activation=hp.Choice('activation',['tanh', 'sigmoid'],
		parent_name='kernel_initializer',
		parent_values=['glorot_uniform', 'glorot_normal'])
	elif kernel_initializer in ['he_uniform', 'he_normal']:
		activation=hp.Choice('activation',['elu','relu'],
		parent_name='kernel_initializer',
		parent_values=['he_uniform', 'he_normal'])
	elif kernel_initializer in ['lecun_uniform','lecun_normal']:
		activation=hp.Choice('activation',['selu'],
		parent_name='kernel_initializer',
		parent_values=['lecun_uniform','lecun_normal'])

	# kernel_regularizer=tf.keras.regularizers.l2(0.01)
	kernel_constraint=tf.keras.constraints.MaxNorm(1.)
	n_neurons=hp.Int('n_neurons',min_value=1,max_value=100,step=5)
	optimizer=hp.Choice('optimizer',['opt1','opt2','opt3','opt4','opt5'])
	# learning_rate=hp.Choice('learning_rate',[0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
	learning_rate=0.2 # decreasing during stable periods, see callback
	loss=hp.Choice('loss',['mse', 'mae'])
	# rate=hp.Choice('rate',[0., 0.1, 0.2, 0.4, 0.6]) # applied on the outputs 
	# dropout=hp.Choice('dropout',[0., 0.1, 0.2, 0.4, 0.6]) # applied on the inputs 
	# recurrent_dropout=hp.Choice('recurrent_dropout',[0., 0.1, 0.2, 0.4, 0.6]) # applied on the recurrent state

	model = tf.keras.Sequential()
	# if activation == 'leakyrelu':
	# 	activation = tf.keras.layers.LeakyReLu(alpha=0.2)
	model.add(tf.keras.layers.LSTM(n_neurons,
											input_shape=(timesteps,n_feature),
											# dropout=dropout,
											# recurrent_dropout=recurrent_dropout,
											kernel_initializer=kernel_initializer,
											activation=activation,
											kernel_constraint=kernel_constraint,
											# kernel_regularizer=kernel_regularizer,
											return_sequences=True,
											stateful=stateful))
										
	# if activation == 'selu':
		# model.add(tf.keras.layers.AlphaDropout(rate=rate))
	# else
	# 	model.add(MCDropout(rate=rate))

	for i in range(hp.Int('num_layers', 0, 3)):
		model.add(tf.keras.layers.LSTM(units=hp.Int('n_neurons' + str(i),
											min_value=1,
											max_value=100,
											step=5),
											# dropout=hp.Choice('dropout' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6]) if activation != 'selu' else 0.,
											# recurrent_dropout=hp.Choice('recurrent_dropout' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6]) if activation != 'selu' else 0.,
											kernel_initializer=kernel_initializer,
											activation=activation,
											# kernel_regularizer=kernel_regularizer,
											kernel_constraint=kernel_constraint,
											return_sequences=True,
											stateful=stateful))

		# if activation == 'selu':
			# model.add(tf.keras.layers.AlphaDropout(rate=hp.Choice('rate' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6])))
		# else
		# 	model.add(MCDropout(rate=hp.Choice('rate' + str(i+1),
		# 									[0., 0.1, 0.2, 0.4, 0.6])))
	model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_neurons)))

	if optimizer == 'opt1':
		optimizer = tf.keras.optimizers.RMSprop(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt2':
		optimizer = tf.keras.optimizers.SGD(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate,nesterov=hp.Choice('nesterov',['True','False']))
	elif optimizer == 'opt3':
		optimizer = tf.keras.optimizers.Adam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt4':
		optimizer = tf.keras.optimizers.Adamax(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt5':
		optimizer = tf.keras.optimizers.Nadam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)

	model.compile(optimizer= optimizer,loss=loss,sample_weight_mode='temporal',metrics=[last_time_step_mse])

	print(model.summary())
	return model
def build_model_GRU(hp, n_feature=n_feature, output_neurons=output_neurons, timesteps = None, stateful = False):

	kernel_initializer=hp.Choice('kernel_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
	if kernel_initializer in ['glorot_uniform', 'glorot_normal']:
		activation=hp.Choice('activation',['tanh', 'sigmoid'],
		parent_name='kernel_initializer',
		parent_values=['glorot_uniform', 'glorot_normal'])
	elif kernel_initializer in ['he_uniform', 'he_normal']:
		activation=hp.Choice('activation',['elu','relu'],
		parent_name='kernel_initializer',
		parent_values=['he_uniform', 'he_normal'])
	elif kernel_initializer in ['lecun_uniform','lecun_normal']:
		activation=hp.Choice('activation',['selu'],
		parent_name='kernel_initializer',
		parent_values=['lecun_uniform','lecun_normal'])


	kernel_constraint=tf.keras.constraints.MaxNorm(1.)
	n_neurons=hp.Int('n_neurons',min_value=1,max_value=100,step=5)
	optimizer=hp.Choice('optimizer',['opt1','opt2','opt3','opt4','opt5'])
	learning_rate=0.2 # decreasing during stable periods, see callback
	loss=hp.Choice('loss',['mse', 'mae'])

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.GRU(n_neurons,
											input_shape=(timesteps,n_feature),
											kernel_initializer=kernel_initializer,
											kernel_constraint=kernel_constraint,
											activation=activation,
											return_sequences=True,
											stateful=stateful))

	for i in range(hp.Int('num_layers', 0, 3)):
		model.add(tf.keras.layers.GRU(units=hp.Int('n_neurons' + str(i),
											min_value=1,
											max_value=100,
											step=5),
											kernel_initializer=kernel_initializer,
											kernel_constraint=kernel_constraint,
											activation=activation,
											return_sequences=True,
											stateful=stateful))

	model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_neurons)))

	if optimizer == 'opt1':
		optimizer = tf.keras.optimizers.RMSprop(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt2':
		optimizer = tf.keras.optimizers.SGD(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate,nesterov=hp.Choice('nesterov',['True','False']))
	elif optimizer == 'opt3':
		optimizer = tf.keras.optimizers.Adam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt4':
		optimizer = tf.keras.optimizers.Adamax(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt5':
		optimizer = tf.keras.optimizers.Nadam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)

	model.compile(optimizer= optimizer,loss=loss,sample_weight_mode='temporal',metrics=[last_time_step_mse])

	print(model.summary())
	return model
def build_model_conv(hp, n_feature=n_feature, output_neurons=output_neurons, timesteps = None):

	kernel_initializer=hp.Choice('kernel_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
	if kernel_initializer in ['glorot_uniform', 'glorot_normal']:
		activation=hp.Choice('activation',['tanh', 'sigmoid'],
		parent_name='kernel_initializer',
		parent_values=['glorot_uniform', 'glorot_normal'])
	elif kernel_initializer in ['he_uniform', 'he_normal']:
		activation=hp.Choice('activation',['elu','relu'],
		parent_name='kernel_initializer',
		parent_values=['he_uniform', 'he_normal'])
	elif kernel_initializer in ['lecun_uniform','lecun_normal']:
		activation=hp.Choice('activation',['selu'],
		parent_name='kernel_initializer',
		parent_values=['lecun_uniform','lecun_normal'])

	loss=hp.Choice('loss',['mse', 'mae'])
	optimizer=hp.Choice('optimizer',['opt1','opt2','opt3','opt4','opt5'])
	learning_rate=0.2 # decreasing during stable periods, see callback
	kernel_constraint=tf.keras.constraints.MaxNorm(axis=[0,1])

	n_layers=hp.Int('n_layers',min_value=1,max_value=4,step=1)
	filters=hp.Int('filters',min_value=1,max_value=15,step=2)
	kernel_size=hp.Int('kernel_size',min_value=2,max_value=6,step=1) #max_value should be nr_features

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=(timesteps,n_feature)))

	for rate in (1,2,4,8)*n_layers:
		model.add(tf.keras.layers.Conv1D(filters=filters,
										kernel_size=kernel_size, 
										padding="causal",
										kernel_initializer=kernel_initializer,
										activation=activation,
										dilation_rate=rate,
										kernel_constraint=kernel_constraint))
		model.add(tf.keras.layers.Conv1D(filters=output_neurons, kernel_size=1))#,kernel_constraint=tf.keras.constraints.MaxNorm(axis=[0,1])))

	if optimizer == 'opt1':
		optimizer = tf.keras.optimizers.RMSprop(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt2':
		optimizer = tf.keras.optimizers.SGD(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate,nesterov=hp.Choice('nesterov',['True','False']))
	elif optimizer == 'opt3':
		optimizer = tf.keras.optimizers.Adam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt4':
		optimizer = tf.keras.optimizers.Adamax(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt5':
		optimizer = tf.keras.optimizers.Nadam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)

	model.compile(optimizer= optimizer,loss=loss, metrics=[last_time_step_mse])

	print(model.summary())
	return model
def build_model_conv1D_LSTM(hp, n_feature=n_feature, output_neurons=output_neurons, timesteps = None, stateful = False):
	
	kernel_initializer=hp.Choice('kernel_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
	if kernel_initializer in ['glorot_uniform', 'glorot_normal']:
		activation=hp.Choice('activation',['tanh', 'sigmoid'],
		parent_name='kernel_initializer',
		parent_values=['glorot_uniform', 'glorot_normal'])
	elif kernel_initializer in ['he_uniform', 'he_normal']:
		activation=hp.Choice('activation',['elu','relu'],
		parent_name='kernel_initializer',
		parent_values=['he_uniform', 'he_normal'])
	elif kernel_initializer in ['lecun_uniform','lecun_normal']:
		activation=hp.Choice('activation',['selu'],
		parent_name='kernel_initializer',
		parent_values=['lecun_uniform','lecun_normal'])

	# kernel_regularizer=tf.keras.regularizers.l2(0.01)
	kernel_constraint=tf.keras.constraints.MaxNorm(1.)

	n_neurons=hp.Int('n_neurons',min_value=1,max_value=100,step=5)
	optimizer=hp.Choice('optimizer',['opt1','opt2','opt3','opt4','opt5'])
	learning_rate=0.2 # decreasing during stable periods, see callback
	loss=hp.Choice('loss',['mse', 'mae'])
	kernel_size=hp.Int('kernel_size',min_value=2,max_value=6,step=2)
	filters=hp.Int('filters',min_value=1,max_value=15,step=2)

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv1D(kernel_size=kernel_size,
											filters=filters,
											kernel_initializer=kernel_initializer,
											activation=activation,
											kernel_constraint=tf.keras.constraints.MaxNorm(axis=[0,1]),
											strides=1,
											padding="same",
											input_shape=(timesteps,n_feature)))
	for i in range(hp.Int('num_layers', 0, 3)):
		model.add(tf.keras.layers.LSTM(units=hp.Int('n_neurons' + str(i),
											min_value=1,
											max_value=100,
											step=5),
											# dropout=hp.Choice('dropout' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6]) if activation != 'selu' else 0.,
											# recurrent_dropout=hp.Choice('recurrent_dropout' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6]) if activation != 'selu' else 0.,
											kernel_initializer=kernel_initializer,
											# kernel_regularizer=kernel_regularizer,
											kernel_constraint=kernel_constraint,
											activation=activation,
											return_sequences=True,
											stateful=stateful))

	model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_neurons)))

	if optimizer == 'opt1':
		optimizer = tf.keras.optimizers.RMSprop(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt2':
		optimizer = tf.keras.optimizers.SGD(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate,nesterov=hp.Choice('nesterov',['True','False']))
	elif optimizer == 'opt3':
		optimizer = tf.keras.optimizers.Adam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt4':
		optimizer = tf.keras.optimizers.Adamax(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt5':
		optimizer = tf.keras.optimizers.Nadam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)

	model.compile(optimizer= optimizer,loss=loss,sample_weight_mode='temporal',metrics=[last_time_step_mse])

	print(model.summary())
	return model
def build_model_conv1D_GRU(hp, n_feature=n_feature, output_neurons=output_neurons, timesteps = None, stateful = False):
		
	kernel_initializer=hp.Choice('kernel_initializer',['glorot_uniform','glorot_normal','he_uniform','he_normal','lecun_uniform','lecun_normal'])
	if kernel_initializer in ['glorot_uniform', 'glorot_normal']:
		activation=hp.Choice('activation',['tanh', 'sigmoid'],
		parent_name='kernel_initializer',
		parent_values=['glorot_uniform', 'glorot_normal'])
	elif kernel_initializer in ['he_uniform', 'he_normal']:
		activation=hp.Choice('activation',['elu','relu'],
		parent_name='kernel_initializer',
		parent_values=['he_uniform', 'he_normal'])
	elif kernel_initializer in ['lecun_uniform','lecun_normal']:
		activation=hp.Choice('activation',['selu'],
		parent_name='kernel_initializer',
		parent_values=['lecun_uniform','lecun_normal'])

	n_neurons=hp.Int('n_neurons',min_value=1,max_value=100,step=5)
	optimizer=hp.Choice('optimizer',['opt1','opt2','opt3','opt4','opt5'])
	learning_rate=0.2 # decreasing during stable periods, see callback
	loss=hp.Choice('loss',['mse', 'mae'])
	kernel_constraint=tf.keras.constraints.MaxNorm(1.)
	kernel_size=hp.Int('kernel_size',min_value=2,max_value=6,step=2)
	filters=hp.Int('filters',min_value=1,max_value=15,step=2)

	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv1D(kernel_size=kernel_size,
											filters=filters,
											kernel_initializer=kernel_initializer,
											activation=activation,
											kernel_constraint=tf.keras.constraints.MaxNorm(axis=[0,1]),
											strides=1,
											padding="same",
											input_shape=(timesteps,n_feature)))
	for i in range(hp.Int('num_layers', 0, 3)):
		model.add(tf.keras.layers.GRU(units=hp.Int('n_neurons' + str(i),
											min_value=1,
											max_value=100,
											step=5),
											# dropout=hp.Choice('dropout' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6]) if activation != 'selu' else 0.,
											# recurrent_dropout=hp.Choice('recurrent_dropout' + str(i+1),
											# [0., 0.1, 0.2, 0.4, 0.6]) if activation != 'selu' else 0.,
											kernel_initializer=kernel_initializer,
											activation=activation,
											# kernel_regularizer=kernel_regularizer,
											kernel_constraint=kernel_constraint,
											return_sequences=True,
											stateful=stateful))

	model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_neurons)))

	if optimizer == 'opt1':
		optimizer = tf.keras.optimizers.RMSprop(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt2':
		optimizer = tf.keras.optimizers.SGD(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate,nesterov=hp.Choice('nesterov',['True','False']))
	elif optimizer == 'opt3':
		optimizer = tf.keras.optimizers.Adam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt4':
		optimizer = tf.keras.optimizers.Adamax(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)
	elif optimizer == 'opt5':
		optimizer = tf.keras.optimizers.Nadam(clipnorm=hp.Choice('clipnorm',[0.5,1.,2.]),
											learning_rate=learning_rate)

	model.compile(optimizer= optimizer,loss=loss,sample_weight_mode='temporal',metrics=[last_time_step_mse])

	print(model.summary())
	return model
