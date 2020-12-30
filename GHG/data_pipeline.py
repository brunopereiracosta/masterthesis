import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from arguments import *

# data loader
def loader(path):
	#['Year', 'Pastagem', 'Cultura+Arável', 'Floresta', 'Urbano/Artificial','Outros', 'GDP', 'Exergy']
	data=pd.read_csv(path)
	# data.index=data.Year
	data=data.drop('Year',1)
	return data
# extract predicted entry on every batch of windows
def Extract(lst): 
		return [item[-1] for item in lst]
# add last category to the matrix
def add_last_category(m):
	col = np.subtract(np.ones((np.matrix(m).shape[0],1))*100,np.matrix(m).sum(axis=1, dtype='float'))
	return np.round(np.append(np.matrix(m), col, axis=1),decimals=1)
# apply lag on data
def lag(dataset, lag_parameter):
	lulc_ghg = dataset.iloc[:,0:n_feature-2]
	exergy_gdp = dataset.iloc[:,n_feature-2:n_feature].shift(-lag_parameter)#, fill_value=0)
	data = pd.concat([lulc_ghg, exergy_gdp], axis=1, sort=False)
	data = data[:-lag_parameter]
	return data
# transform flat data into batch (nr batch x window_size x features)
def batching_data(dataset, target, window_size=window_size, step=1):
	data, labels = [],[]

	start_index = window_size
	end_index = len(dataset)

	for i in range(start_index, end_index):
		indices = range(i-window_size, i, step)
		indices_2 = range(i - window_size+1, i+1, step)
		data.append(dataset[indices])
		labels.append(target[indices_2])

	#return tf.convert_to_tensor(data), tf.convert_to_tensor(labels)
	return np.array(data), np.array(labels)
# data division into n_splits-folds for cross-validation and standardisation (=! normalisation)
# fold -> [0,4], 5th fold (fold=4) for training with all data
def split_norm_batch(data,fold_selector,nr_folds=5):

	x = np.array(data.values[:,0:n_feature])
	y = np.array(data.values[:,4:6])
	# exit()

	x_train_folds, x_train_norm = [], []
	x_val_folds, x_val_norm = [], []
	x_test_folds, x_test_norm = [], []
	y_train_folds, y_train_norm = [], []
	y_val_folds, y_val_norm = [], []
	y_test_folds, y_test_norm = [], []
	scalers_2 = []


	for f in range(nr_folds):
		if f==4:
			x_train_folds.append(pd.DataFrame(x[list(range(0,49))]))
			x_val_folds.append(pd.DataFrame(x[list(range(43,55))]))
			x_test_folds.append([])

			y_train_folds.append(pd.DataFrame(y[list(range(0,49))]))
			y_val_folds.append(pd.DataFrame(y[list(range(43,55))]))
			y_test_folds.append([])
		else:
			x_train_folds.append(pd.DataFrame(x[list(range(0,37+f*2))]))
			x_val_folds.append(pd.DataFrame(x[list(range(31+f*2,43+f*2))]))
			x_test_folds.append(pd.DataFrame(x[list(range(37+f*2,49+f*2))]))

			y_train_folds.append(pd.DataFrame(y[list(range(0,37+f*2))]))
			y_val_folds.append(pd.DataFrame(y[list(range(31+f*2,43+f*2))]))
			y_test_folds.append(pd.DataFrame(y[list(range(37+f*2,49+f*2))]))

		# scaler1 = StandardScaler()
		scaler1 = MinMaxScaler(feature_range=(0, 1))
		# scaler2 = StandardScaler()
		scaler2 = MinMaxScaler(feature_range=(0, 1))
		scalers_2.append(scaler2)
		scaler_x = scaler1.fit(pd.DataFrame(x_train_folds[f]))
		scaler_y = scaler2.fit(pd.DataFrame(y_train_folds[f]))

		x_train_norm.append(scaler_x.transform(x_train_folds[f]))
		x_val_norm.append(scaler_x.transform(x_val_folds[f]))

		y_train_norm.append(scaler_y.transform(y_train_folds[f]))
		y_val_norm.append(scaler_y.transform(y_val_folds[f]))
	
		if f==4:
			x_test_norm.append([])
			y_test_norm.append([])
		else:
			x_test_norm.append(scaler_x.transform(x_test_folds[f]))
			y_test_norm.append(scaler_y.transform(y_test_folds[f]))

	x_train, y_train = batching_data(dataset=x_train_norm[fold_selector],target=y_train_norm[fold_selector])
	x_val, y_val = batching_data(dataset=x_val_norm[4],target=y_val_norm[fold_selector])
	x_test, y_test = batching_data(dataset=x_test_norm[fold_selector],target=y_test_norm[fold_selector])

	return x_train, y_train, x_val, y_val, x_test, y_test, scalers_2[fold_selector], scaler_x