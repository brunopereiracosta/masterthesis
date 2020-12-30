import tensorflow as tf
import kerastuner as kt
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import t
import keras
import numpy as np
from data_pipeline import split_norm_batch
from models_bayes import Extract, last_time_step_mse
import pandas as pd
from score_iterator import *
from data_pipeline import loader
from visualisation import add_last_category, deserialization
from arguments import *



def forecast(model,data,fold_selector,path,directory,baseline,scenario):
	#loading scenarios data
	if scenario == 'lince':
		scenarios = loader('/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/lince_ghg.csv')
		# scenarios = loader('/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/high_gdp.csv')
	elif scenario == 'avestruz':
		scenarios = loader('/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/avestruz_ghg.csv')
		# scenarios = loader('/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/low_gdp.csv')
	else:
		print("Scenario unavailable")
		return

	x_train, y_train, x_val, y_val, x_test, y_test, scaler2, scaler_x = split_norm_batch(data,fold_selector)

	# Bayesian search
	tuner = kt.BayesianOptimization(model,
					objective = kt.Objective('val_last_time_step_mse',direction="min"),
					max_trials=2000,
					num_initial_points=300,
					directory = directory,
					project_name = "fold_"+str(fold_selector),
					seed=81451)
	# best 3 models
	tuner.reload()
	models = tuner.get_best_models(num_models=3)

	path_1 = '/Volumes/PEN/LULC/'
	path1 = path_1 + 'Simple_15_11_2020_logs/'
	path2 = path_1 + 'LSTM_15_11_2020_logs/'
	path3 = path_1 + 'GRU_15_11_2020_logs/'
	path4 = path_1 + 'conv_15_11_2020_logs/'
	path5 = path_1 + 'conv1D_LSTM_15_11_2020_logs/'
	path6 = path_1 + 'conv1D_GRU_15_11_2020_logs/'
 
	study='_lulc'
	paths = [path1, path2, path3, path4, path5, path6]
	m_predict = [[]for i in range(len(paths))]
	m_predict_average = [[]for i in range(len(paths))]
	for i, path_2 in enumerate(paths):
		m_predict[i] = deserialization(path_2,fold_selector,scenario,study,True)
		# print(m_predict[i])
		m_predict_average[i] = np.round(sum(m_predict[i])/len(m_predict[i]),1)
		# print(m_predict_average[i])
		m_predict_average[i][:, [0,1,2,3,4]] = m_predict_average[i][:, [4,0,1,2,3]]
		# print(m_predict_average[i])
	m_predict_average_total = np.round(sum(m_predict_average)/len(m_predict_average),1)
	# print(m_predict_average_total)
	# m = np.insert(m_predict_average_total,0,lulc_2016,axis=0)
	# print(m)
	m_lulc = m_predict_average_total[:, [1,2,3,4]]
	# m1 é a matriz a entrar no modelo, começa em 2016 (LULC)
	# print(m_lulc)
	# print(m_lulc[0,:])

	lulc_2015 = [19.5, 36.2, 5.0, 19]
	ghg_2015 = [6419.8, 49995.1]
	lulc_2016 = [19.3, 36.3, 5.0, 18.8]
	ghg_2016 = [6382.8, 50362.3]
	mses = {}
	y_preds_extracted_inverted_total = []
	for i in range(0,3): # 3 best models
		x_new = []
		x_new_tmp = []
		y_preds = []
		y_preds_extracted = []
		y_preds_extracted_inverted = []
		print(i)
		mses.update({'MSE'+str(i) : round(models[i].evaluate(x_val, y_val)[1],3)})
		for j in range(len(scenarios)):
			print(j)

			# prediction from best 3 models
			if j == 0:
				# batch_to_predict_tmp = x_val[-1][np.newaxis,...]
				# x_val[-1] vai até 2014 pq causa do batching, só o y_val[-1] vai até 2015
				# scaler_x ≠ scaler_y, é necessário utilizar x_val
				batch_to_predict_tmp = x_val[-1][1:,]
				x_new_tmp.append([np.concatenate((lulc_2015,ghg_2015,scenarios.values[j,:]))])
				x_new.append([scaler_x.transform(x_new_tmp[j])[0][0:n_feature]])
				batch_to_predict = np.append(batch_to_predict_tmp,x_new[j],axis=0)[np.newaxis,...]
				
			elif j == 1:
				batch_to_predict_tmp = batch_to_predict[0][1:,]
				x_new_tmp.append([np.concatenate((lulc_2016,ghg_2016,scenarios.values[j,:]))])
				x_new.append([scaler_x.transform(x_new_tmp[j])[0][0:n_feature]])
				batch_to_predict = np.append(batch_to_predict_tmp,x_new[j],axis=0)[np.newaxis,...]
				
				y_preds.append(models[i].predict(batch_to_predict))
				y_preds_extracted.append(Extract(y_preds[j-1]))
				y_preds_extracted_inverted.append(scaler2.inverse_transform(y_preds_extracted[j-1])[0])
				print(y_preds_extracted_inverted)

			else:
				x_new_tmp.append([np.concatenate((m_lulc[j,:],Extract(y_preds[j-2])[0],scenarios.values[j,:]))])

				# x scaling is only considered in the new data
				x_new.append([np.concatenate((scaler_x.transform(x_new_tmp[j])[0][0:4],
												x_new_tmp[j][0][4:4+output_neurons],
												scaler_x.transform(x_new_tmp[j])[0][4+output_neurons:n_feature]))])

				batch_to_predict = np.append(batch_to_predict[0][1:],x_new[j],axis=0)[np.newaxis,...]

				y_preds.append(models[i].predict(batch_to_predict))
				y_preds_extracted.append(Extract(y_preds[j-1]))
				y_preds_extracted_inverted.append(scaler2.inverse_transform(y_preds_extracted[j-1])[0])
				print(y_preds_extracted_inverted)

			
		y_preds_extracted_inverted_total.append(y_preds_extracted_inverted)
		print(y_preds_extracted_inverted_total)
		# exit()

	print(mses)
	print('####### Baseline Metrics #######')
	# mse_baseline = baseline_model(x_train, y_train, x_val, y_val, data, epochs)
	mse_baseline = baseline
	print(mse_baseline)

	print('fold_selector - ',fold_selector)

	model_name = str(model)[str(model).find(" ")+1:str(model).find(" ",str(model).find(" ")+1)]
	
	context = {'Model': model_name, 'fold' : fold_selector}
	statistics = {'MSE' : mses, 'MSE_baseline' : mse_baseline}
	targets = {'Predicted Values' : y_preds_extracted_inverted_total}
	fused_data = {'Context' : context, 'Statistics' : statistics,'Targets' : targets}

	# with open('/Volumes/PEN/' + directory +'_logs/' + 'fold_' + str(fold_selector) + '_' + scenario + '_ghg' + '.json', 'w', encoding='utf-8') as f:
	with open(path+'_logs/'+'fold_'+ str(fold_selector) + '_' + scenario + '_ghg_gdp' + '.json', 'w', encoding='utf-8') as f:
		# json.dump(statistics, f, ensure_ascii=False, indent=4)
		json.dump(fused_data, f, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)