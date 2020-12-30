import tensorflow as tf
import kerastuner as kt
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import t
import keras
import os
import random
import numpy as np
from data_pipeline import split_norm_batch
from models_bayes import baseline_model, Extract, last_time_step_mse
import pandas as pd
from score_iterator import *
import datetime
import csv

# for reproducibility
def reset_random_seeds():
	seed_value= 81451
	os.environ['PYTHONHASHSEED']=str(0)
	np.random.seed(seed_value)
	random.seed(seed_value)
	tf.random.set_seed(seed_value)

# add last category to the matrix
def add_last_category(m):
	col = np.subtract(np.ones((np.matrix(m).shape[0],1))*100,np.matrix(m).sum(axis=1, dtype='float'))
	return np.round(np.append(np.matrix(m), col, axis=1),decimals=1)

# column-wise RMSE
def columnwiseRMSE(m1,m2):
	rmse = []
	if np.matrix(m1).shape != np.matrix(m2).shape:
		raise Exception('Cannot calculate RMSE, matrices of different dimensions: {} vs {}'.format(m1.shape,m2.shape))
	for l in range(np.matrix(m1).shape[1]):
		rmse.append(np.sqrt(mean_squared_error(np.matrix(m1)[:,l],np.matrix(m2)[:,l])))
	return rmse

# mainpath = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/'
def optimiser(model,data,fold_selector,path,directory,baseline):
	x_train, y_train, x_val, y_val, x_test, y_test, scaler2, scaler_x = split_norm_batch(data,fold_selector)

	# Bayesian search
	tuner = kt.BayesianOptimization(model,
					objective = kt.Objective('val_last_time_step_mse',direction="min"),
					max_trials=2000,
					num_initial_points=300,
					directory = directory,
					project_name = "fold_"+str(fold_selector),
					seed=81451)#,
					#max_model_size=1e5)
					# alpha=1e-4,
					# beta=3)
	# callbacks
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_last_time_step_mse', 
										factor=0.5, patience=6, min_lr=0.001,mode='min')
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_last_time_step_mse',
															patience=14, mode='min')
	log_dir = path + '_logs/' + "fold_"+str(fold_selector) + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	
	# print(path + '_logs/' + "fold_"+str(fold_selector) + '/')

	# search
	epochs = 1000
	tuner.search(x_train, y_train, epochs=epochs, validation_data=(x_val,y_val),
				callbacks=[early_stop, reduce_lr], verbose=0)
	
	# save best 3 models
	models = tuner.get_best_models(num_models=3)

	# hists = []
	mses = {}
	final_rmses_tmp = []
	final_rmses = {}
	y_preds = []
	y_preds_extracted = []
	y_preds_extracted_inverted = []
	if fold_selector != 4:
		y_test_extracted = Extract(y_test)
		y_test_extracted_inverted = scaler2.inverse_transform(y_test_extracted)
	for i in range(0,3): # 3 best models
		# hists.append(models[i].fit(x_train, y_train, epochs=epochs, shuffle=False, validation_data=(x_val, y_val),
		# 					callbacks=[early_stop,reduce_lr],verbose=0))
		mses.update({'MSE'+str(i) : round(models[i].evaluate(x_val, y_val)[1],3)})
		# prediction from best 3 models
		if fold_selector != 4:
			y_preds.append(models[i].predict(x_test)) # y_pred=final_predictions
			y_preds_extracted.append(Extract(y_preds[i]))
			y_preds_extracted_inverted.append(scaler2.inverse_transform(y_preds_extracted[i]))
			final_rmses_tmp = columnwiseRMSE(y_test_extracted_inverted,y_preds_extracted_inverted[i])
			# final_rmses_tmp.append(np.sqrt(np.sum(np.square(final_rmses_tmp), axis=0)))
			final_rmses.update({'RMSE'+str(i) : np.round(final_rmses_tmp,decimals=3)})
	print(mses)
	
	print('####### Baseline Metrics #######')
	# mse_baseline = baseline_model(x_train, y_train, x_val, y_val, data, epochs)
	mse_baseline = baseline
	print(mse_baseline)
	
	if fold_selector != 4:
		print('####### Real Values #######')
		print(y_test_extracted_inverted)
		print('####### Predicted Values #######')
		print(y_preds_extracted_inverted[0])
		print(y_preds_extracted_inverted[1])
		print(y_preds_extracted_inverted[2])

		print('####### Final Root Mean Squared Errors #######')
		print(final_rmses)

	print('fold_selector - ',fold_selector)

	model_name = str(model)[str(model).find(" ")+1:str(model).find(" ",str(model).find(" ")+1)]
	
	context = {'Model': model_name, 'fold' : fold_selector}
	if fold_selector != 4:
		statistics = {'MSE' : mses, 'MSE_baseline' : mse_baseline, 'Final_RMSE' : final_rmses}
		targets = {'Real Values' : y_test_extracted_inverted, 'Predicted Values' : y_preds_extracted_inverted}
		fused_data = {'Context' : context, 'Statistics' : statistics,'Targets' : targets}
	else:
		statistics = {'MSE' : mses, 'MSE_baseline' : mse_baseline}
		fused_data = {'Context' : context, 'Statistics' : statistics}

	with open(path+'_logs/'+'fold_'+ str(fold_selector) + '.json', 'w', encoding='utf-8') as f:
		# json.dump(statistics, f, ensure_ascii=False, indent=4)
		json.dump(fused_data, f, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)