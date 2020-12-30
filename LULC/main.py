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
data = loader('/Users/bpc/6º Ano-2º Sem./Tese/Uso do Solo/Dados/Dados Finais.csv')
# print(data)
# lagged LULC data by t periods
data = lag(data,1) # currently hard coded to 1
# print(data)
# exit()

# Baseline calculation
def mse_baseline():
	print("HERE")
	mse_baseline = []
	for i in range(5):
		print(i)
		x_train, y_train, x_val, y_val, x_test, y_test, scaler2, scaler_x = split_norm_batch(data,fold_selector=i)
		epochs = 1000
		mse_baseline.append(baseline_model(x_train, y_train, x_val, y_val, data, epochs))
	return mse_baseline

fold_selector=4
baseline = mse_baseline()

# directory = 'LULC/GRU_4_11_2020'
# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/' + directory
# path = '/Volumes/PEN/' + directory

# optimiser(build_model_GRU,data,fold_selector,path,directory,baseline[fold_selector])


######## FORECASTING #########
scenarios=['avestruz','lince']
directory1 = 'LULC/Simple_15_11_2020'
model1 = build_model_Simple
directory2 = 'LULC/LSTM_15_11_2020'
model2 = build_model_LSTM
directory3 = 'LULC/GRU_15_11_2020'
model3 = build_model_GRU
directory4 = 'LULC/conv_15_11_2020'
model4 = build_model_conv
directory5 = 'LULC/conv1D_LSTM_15_11_2020'
model5 = build_model_conv1D_LSTM
directory6 = 'LULC/conv1D_GRU_15_11_2020'
model6 = build_model_conv1D_GRU
directories = (directory1,directory2,directory3,directory4,directory5,directory6)
models = (model1,model2,model3,model4,model5,model6)

for scenario in scenarios:
	for direc, model in zip(directories,models):
		path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/' + direc
		forecast(model,data,4,path,direc, baseline=0.6413,scenario=scenario)


##### 
# plotTrainingErrors(fold_selector,True)
#####
# data = loader('/Users/bpc/6º Ano-2º Sem./Tese/Uso do Solo/Dados/Dados Finais.csv').to_numpy()
# data = swap_columns(data)
# plotData(data,True,True)
#####
# plotForecast(show=True,save=False,scenario='lince')
#####



# tensorboard --logdir path + "logs/fold_"+str(fold_selector) + '/'