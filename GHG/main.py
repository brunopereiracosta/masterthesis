# import sys, os
# myfolder = os.path.dirname(os.path.abspath(__file__)) + "/"
# sys.path.append(myfolder + "../../../Analysis")
from bayes_search import * 
from score_iterator import *
from data_pipeline import *
from models_bayes import *
from arguments import *
from visualisation import *
from forecaster import *

# for reproducibility
reset_random_seeds()

#loading data
data = loader('/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/Dados Finais.csv')
# lagged LULC data by t periods
data = lag(data,1) # currently hard coded to 1

# Baseline calculation
def mse_baseline():
	mse_baseline = []
	for i in range(5):
		x_train, y_train, x_val, y_val, x_test, y_test, scaler2, scaler_x = split_norm_batch(data,fold_selector=i)
		epochs = 1000
		mse_baseline.append(baseline_model(x_train, y_train, x_val, y_val, data, epochs))
		print(mse_baseline)
	return mse_baseline


# fold_selector=4
# baseline = mse_baseline()
# directory = 'Simple_1_12_2020'
# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/' + directory

# directory = '/Volumes/PEN/GHG/conv1D_GRU_20_11_2020'
# path = directory

# optimiser(build_model_Simple,data,fold_selector,path,directory,baseline[fold_selector])

######## FORECASTING #########
scenarios=['avestruz','lince']
directory1 = 'Simple_1_12_2020'
model1 = build_model_Simple
directory2 = 'LSTM_1_12_2020'
model2 = build_model_LSTM
directory3 = 'GRU_1_12_2020'
model3 = build_model_GRU
directory4 = 'conv_2_12_2020'
model4 = build_model_conv
directory5 = 'conv1D_LSTM_2_12_2020'
model5 = build_model_conv1D_LSTM
directory6 = 'conv1D_GRU_2_12_2020'
model6 = build_model_conv1D_GRU
directories = (directory1,directory2,directory3,directory4,directory5,directory6)
models = (model1,model2,model3,model4,model5,model6)

for scenario in scenarios:
	for direc, model in zip(directories,models):
		path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/' + direc
		forecast(model,data,4,path,direc, baseline=0.6413,scenario=scenario)

# for scenario in scenarios:
# 	path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/' + directory4
# 	forecast(model4,data,fold_selector,path,directory4, baseline=0.6413,scenario=scenario)

# plotTrainingErrors(fold_selector,True)
# plotData(data)
# plotForecast(show=True,save=True)

# tensorboard --logdir path + "logs/fold_"+str(fold_selector) + '/'