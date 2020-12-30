import json
import numpy as np
from sklearn.metrics import mean_squared_error
from data_pipeline import add_last_category


# column-wise RMSE
def columnwiseRMSE(m1,m2):
	rmse = []
	if np.matrix(m1).shape != np.matrix(m2).shape:
		raise Exception('Cannot calculate RMSE, matrices of different dimensions: {} vs {}'.format(m1.shape,m2.shape))
	for l in range(np.matrix(m1).shape[1]):
		rmse.append(np.sqrt(mean_squared_error(np.matrix(m1)[:,l],np.matrix(m2)[:,l])))
	return rmse

# Deserialization
def deserialization(path,fold_selector,apply_criteria=False):
	with open(path +'fold_' + str(fold_selector) + '.json', "r") as r:
		dics = json.load(r)
		cont = dics['Context']
		stats = dics['Statistics']
		y_test_extracted_inverted = np.asarray(dics['Targets']['Real Values'])
		y_preds_extracted_inverted = np.asarray(dics['Targets']['Predicted Values'])
		rmses = stats['Final_RMSE']
		print(cont)
		# print(rmses)
		# print(stats)
		# print(y_test_extracted_inverted)
		# print(y_preds_extracted_inverted)
		if apply_criteria:
			all_rmse = []
			sum_rmse = []
			for rmse in rmses.values():
				all_rmse.append(rmse)
			for i in range(len(all_rmse)):
				sum_rmse.append(np.round(np.sum(all_rmse[i]),1))
			print(sum_rmse)
			indexes = [i for i,v in enumerate(sum_rmse) if v > 15]
			# print(indexes)
			for index in sorted(indexes, reverse=True):
				y_preds_extracted_inverted = np.delete(y_preds_extracted_inverted,index,axis=0)
				del rmses['RMSE'+str(index)]
			# sum_rmse[:] = [x for x in sum_rmse if x<15]

		# if fold_selector == 0 and \
		# (path == "/Users/bpc/6º Ano-2º Sem./Tese/Coding/conv_7_11_2020_logs/" or \
		# path == "/Users/bpc/6º Ano-2º Sem./Tese/Coding/conv1D_LSTM_7_11_2020_logs/"):
		# 	y_pred_curated = np.delete(y_preds_extracted_inverted,2,axis=0)
		# 	return y_test_extracted_inverted, y_pred_curated, rmses
		
		# if fold_selector == 1 and \
		# path == "/Users/bpc/6º Ano-2º Sem./Tese/Coding/conv1D_LSTM_7_11_2020_logs/":
		# 	y_pred_curated = np.delete(y_preds_extracted_inverted,0,axis=0)
		# 	del rmses['RMSE0']
		# 	# print(rmse)
		# 	# rmse_curated = np.delete(rmse,0,0)
		# 	return y_test_extracted_inverted, y_pred_curated, rmses
		return y_test_extracted_inverted, y_preds_extracted_inverted, rmses

# fold_selector = 0
# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/LULC_repeat/'
path = '/Volumes/PEN/LULC/'
path1 = path + 'Simple_1_11_2020_logs/'
path2 = path + 'LSTM_4_11_2020_logs/'
path3 = path + 'GRU_4_11_2020_logs/'
path4 = path + 'conv_7_11_2020_logs/'
path5 = path + 'conv1D_LSTM_7_11_2020_logs/'
path6 = path + 'conv1D_GRU_7_11_2020_logs/'
paths = [path1, path2, path3, path4, path5, path6]

ground_error = []
for fold_selector in range(4):
	m_true = []
	m_predict = [[]for i in range(len(paths))]
	m_predict_average = []
	rmses = [[]for i in range(len(paths))]
	for i, path in enumerate(paths):
		m_true, m_predict[i], rmses[i] = deserialization(path,fold_selector,True)
		if len(m_predict[i]) == 0:
			print("********* No valid SM in:", path,"*********")
			continue
		print(m_predict[i])
		print(len(m_predict[i]))
		m_predict_average.append(np.round(sum(m_predict[i])/len(m_predict[i]),1))

	all_rmse = []
	for i in range(len(paths)):
		for rmse in rmses[i].values():
			all_rmse.append(rmse)
	# print(all_rmse)
	# exit()

	print("********** 1st (intra) analysis **********")
	print("Average ALL RMSES")
	print(np.round(np.sum(all_rmse,axis=0)/len(all_rmse),1))
	print("Average ALL RMSES Collpased")
	print(np.round(np.sum(all_rmse)/len(all_rmse),1))

	print('Nr of SMs:',len(all_rmse))
	# print(all_rmse)
	print("All RMSES Collpased (threshold=15%)")
	for i in range(len(all_rmse)):
		print(np.round(np.sum(all_rmse[i]),1))
	# exit()


	# RMSE quando se faz a média dos resultados dos sub-modelos para cada modelo
	a = []
	# RMSEs calculados gerando nova matriz de previsão
	for i in range(len(m_predict_average)):
		if len(m_predict_average[i]) == 0:
			print("********* Model not included:", paths[i],"*********")
			continue
		# print(m_predict_average[i])
		a.append(columnwiseRMSE(m_true,m_predict_average[i]))
	# RMSEs calculados fazendo a média dos RMSEs de cada modelo
	# counter = 0
	# tmp = 0
	# for k in range(len(paths)):
	# 	if counter > 0:
	# 		a.append(np.round(np.sum(all_rmse[tmp:tmp+len(m_predict[k])],axis=0)/len(all_rmse[tmp:tmp+len(m_predict[k])]),3))
	# 	else:
	# 		a.append(np.round(np.sum(all_rmse[0:len(m_predict[k])],axis=0)/len(all_rmse[0:len(m_predict[k])]),3))
	# 	counter += 1
	# 	tmp += len(m_predict[k])

	# SM aggregated by M
	print("********** 2nd analysis **********")
	print("RMSEs for the 6 models")
	for i in range(len(a)):
		print(np.round(a[i],1))
		print(np.round(np.sum(a[i],axis=0),1))
	print("Average error for the 6 models:")
	print(np.round(np.sum(a,axis=0)/len(a),1))
	ground_error.append(np.round(np.sum(a,axis=0)/len(a),1))
	print(np.sum(np.round(np.sum(a,axis=0)/len(a),1)))

	# m_predict_average_total = np.round(sum(m_predict_average)/len(m_predict_average),1)

	# # RMSE global quando se faz a média global dos resultados de todos os modelos
	# print("RMSES averaged Ms Collpased")
	# print(np.round(np.sum(columnwiseRMSE(m_true,m_predict_average_total))))

print("Final Error")
print(np.round(np.sum(ground_error,axis=0)/len(ground_error),1))
print(np.round(np.sum(np.round(np.sum(ground_error,axis=0)/len(ground_error),1)),1))