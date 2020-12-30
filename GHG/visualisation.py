import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import mean_squared_error
from data_pipeline import loader, lag
# from rmse import deserialization
# import scipy
# import scipy.sparse as sp
# import scipy._lib
# from validation import check_array, check_consistent_length
# from regression import mean_squared_error


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

# Deserialization
def deserialization(path,fold_selector,scenario,study,apply_criteria=False):
	with open(path +'fold_' + str(fold_selector) + '_' + scenario + study + '.json', "r") as r:
		dics = json.load(r)
		cont = dics['Context']
		stats = dics['Statistics']
		if fold_selector != 4:
			y_test_extracted_inverted = np.asarray(dics['Targets']['Real Values'])
			rmses = stats['Final_RMSE']
		y_preds_extracted_inverted = np.asarray(dics['Targets']['Predicted Values'])
		print(cont)
		# print(rmses)
		# print(stats)
		# print(y_test_extracted_inverted)
		# print(y_preds_extracted_inverted)
		if apply_criteria and fold_selector != 4:
			all_rmse = []
			sum_rmse = []
			for rmse in rmses.values():
				all_rmse.append(rmse)
			for i in range(len(all_rmse)):
				sum_rmse.append(np.round(np.sum(all_rmse[i]),1))
			# print(sum_rmse)
			indexes = [i for i,v in enumerate(sum_rmse) if v > 15]
			# print(indexes)
			for index in sorted(indexes, reverse=True):
				y_preds_extracted_inverted = np.delete(y_preds_extracted_inverted,index,axis=0)
				del rmses['RMSE'+str(index)]
			# sum_rmse[:] = [x for x in sum_rmse if x<15]

		if fold_selector != 4:
			return y_test_extracted_inverted, y_preds_extracted_inverted, rmses
		else:
			return y_preds_extracted_inverted

def swap_columns(your_list):
	n_col = len(your_list[0])
	for item in your_list:
		print(item)
		item[n_col-n_col], item[n_col-n_col+1], item[n_col-n_col+2], item[n_col-n_col+3], item[n_col-n_col+4] = \
		item[n_col-1], item[n_col-n_col], item[n_col-n_col+1], item[n_col-n_col+2], item[n_col-n_col+3]
	return your_list


# final_rmses_tmp = []
# final_rmses = {}
# for i in range(0,3):
#	final_rmses_tmp = columnwiseRMSE(y_test_extracted_inverted,y_preds_extracted_inverted[i])
#	final_rmses.update({'RMSE'+str(i) : np.round(final_rmses_tmp,decimals=3)})

# print(sum(final_rmses_tmp)/len(final_rmses_tmp))
# print(final_rmses)

def plotTrainingErrors(fold_selector,show=True,save=False):
	MEDIUM_SIZE = 12
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/'
	path = '/Volumes/PEN/GHG/'
	path1 = path + 'Simple_1_11_2020_logs/'
	path2 = path + 'LSTM_4_11_2020_logs/'
	path3 = path + 'GRU_4_11_2020_logs/'
	path4 = path + 'conv_7_11_2020_logs/'
	path5 = path + 'conv1D_LSTM_7_11_2020_logs/'
	path6 = path + 'conv1D_GRU_7_11_2020_logs/'

	fold_dates = []
	for l in range(4):
		fold_dates.append(np.arange(1999+l,2004+l,1))

	paths = [path1, path2, path3, path4, path5, path6]
	m_true = [[]for i in range(len(paths))]
	m_predict = [[]for i in range(len(paths))]
	m_predict_average = [[]for i in range(len(paths))]
	for i, path in enumerate(paths):
		m_true[i], m_predict[i], rmse = deserialization(path,fold_selector,True)
		print(m_true[i])
		m_true[i][:, [0,1,2,3,4]] = m_true[i][:, [4,0,1,2,3]]
		# m_true[i] = swap_columns(m_true[i])
		print(m_true[i])
		# exit()
		# print(m_predict[i])
		m_predict_average[i] = np.round(sum(m_predict[i])/len(m_predict[i]),1)
		print(m_predict_average[i])
		m_predict_average[i][:, [0,1,2,3,4]] = m_predict_average[i][:, [4,0,1,2,3]]
		print(m_predict_average[i])
		# exit()
		# m_predict_average[i] = swap_columns(m_predict_average[i])


	fig, axs = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, 
				            figsize=(8,11))
	fig.subplots_adjust(left=0.125, bottom=0.2, right=0.95, top=0.95, wspace=None, hspace=None)
    
	#1st graph (1,1) (row,col)
	#2nd graph (2,1)
	#3rd graph (3,1)
	#4th graph (1,2)
	#5th graph (2,2)
	#6th graph (3,2)

	counter = 0
	colors = ['royalblue','darkorange','forestgreen','black','firebrick']
	for j in range(2):
		for k in range(3):
			for i in range(len(m_true[0])):
				axs[k][j].plot(fold_dates[fold_selector],m_true[0][:,i],marker="o",
														linestyle="solid",
														# color='rgbkm'[i],
														color=colors[i],
														linewidth=0.6,alpha=1,markersize=2)#,
														#label="M1")
				axs[k][j].plot(fold_dates[fold_selector],m_predict_average[counter][:,i],
														marker="x",
														linestyle="dashdot",
														# color='rgbkm'[i],
														color=colors[i],
														linewidth=0.6,alpha=0.7,markersize=5),
														#label="SM"+str(i))
				axs[k][j].title.set_text("M"+str(counter+1))
			counter += 1
    
	categories = [ 'Perm. Pasture', 'Perm. Culture & Arable Land', 'Forest', 'Urban/Artificial', 'Other']
	styles = ["solid", "dashdot"]
	markers = ["o","x"]
	labels = ['True Values', 'Predicted Values']
	counter = 0
	for j in range(2):
		for k in range(3):
			for i in range(len(m_true[0])):
				axs[k][j].plot(np.NaN, np.NaN, color=colors[i], label=categories[i])
				axs[k][j].plot(fold_dates[fold_selector],m_predict_average[counter][:,i],
														marker="x",
														linestyle="dashdot",
														color=colors[i],
														linewidth=0.6,markersize=2),
														#label="SM"+str(i))
				axs[k][j].title.set_text("M"+str(counter+1))
			counter += 1
	ax2 = axs[2][1].twinx()
	for m in range(len(styles)):
		ax2.plot(np.NaN, np.NaN, color='gray',linestyle=styles[m],marker=markers[m],markersize=5,
				label=labels[m])
    
	# lines = axs[2][1].get_lines()
	# legend1 = plt.legend([lines[i] for i in [0,2,4,6,8]], categories, loc=1)
	# legend2 = plt.legend([lines[i] for i in [0,3,6]], ['2','v'], loc=4)
	# axs[2][1].add_artist(legend1)
	# axs[2][1].add_artist(legend2)
	# as2 = axs[0][0].twinx()

	ax2.get_yaxis().set_visible(False)
	axs[2][1].legend(bbox_to_anchor=(0.94, -0.65), loc='lower right')
	ax2.legend(bbox_to_anchor=(-1., -0.4), loc='lower left')

	plt.setp(axs[1, 0], ylabel='Land Use Distribution (%)')
	plt.setp(axs[1, 0], xlabel='Year')
	axs[1][0].yaxis.set_label_coords(-0.2, 0.5)
	axs[1][0].xaxis.set_label_coords(-0.04, -1.3)
	axs[1][0].set_yticks(np.arange(0,36,5))
	# plt.yticks(np.arange(0,36,2))

	# fig.tight_layout()
	# fig.suptitle("Fold 0",y=1.5)
	if show:
		plt.show()
		if save == True:
			fig.savefig('/Users/bpc/floobits/bpc/Dissertation_Bruno/Figures/test_fold_'+str(fold_selector)+'.pdf', format='pdf', dpi=3600)
		return
	plt.close()
	return

def plotData(data,show=True,save=False):
	SMALL_SIZE = 8
	MEDIUM_SIZE = 12
	BIGGER_SIZE = 18

	plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=14.5)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=14.5)    # fontsize of the tick labels
	plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	ncols=2
	nrows=1

	# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=False, 
	# 					figsize=(8,11))
	###############################################################################

	fig, (ax,ax1) = plt.subplots(ncols=ncols, nrows=nrows, sharex=False, sharey=False, 
					            figsize=(16,5))
	ax.grid(linestyle='dotted')
	ax1.grid(linestyle='dotted')
	fig.subplots_adjust(left=0.06, bottom=0.12, right=0.994, top=0.95, wspace=0.25, hspace=None)
	
	fold_dates = np.arange(1961,2017,1)
	# print(fold_dates)
	# print(len(fold_dates))
	# print(data[:,0:5])
	# exit()
	colors = ['blue','darkgreen']
	# labels = ['Scenario $\it{P}$','Scenario $\it{O}$']
	ax.plot(fold_dates,data[:,4],
			marker="o",
			linestyle="solid",
			color=colors[0],
			linewidth=1.25,alpha=1,markersize=3),
			#label="SM"+str(i))

	ax2 = ax.twinx()
	ax2.plot(fold_dates,data[:,5],
			marker="o",
			linestyle="solid",
			color=colors[1],
			linewidth=1.25,alpha=1,markersize=3)
			#label="SM"+str(i))

	ax1.plot(fold_dates,data[:,7],
			marker="o",
			linestyle="solid",
			color="black",
			linewidth=1.25,alpha=1,markersize=3),
			#label="SM"+str(i))

	ax.tick_params(axis='y', labelcolor=colors[0])
	ax2.tick_params(axis='y', labelcolor=colors[1])
	# ax.set_xticks(np.arange(1961,2017,9))
	ax.set_xticks(np.arange(1961,2017,6))
	ax1.set_xticks(np.arange(1961,2017,6))
	ax.set_yticks(np.arange(5900,8000,200))
	ax2.set_yticks(np.arange(7000,71000,7000))
	ax.ticklabel_format(style='sci',scilimits=(1,2),axis='y')
	ax2.ticklabel_format(style='sci',scilimits=(2,4),axis='y')
	ax1.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')

	# # for m in range(len(linestyles)):
	# # 	ax2.plot(np.NaN, np.NaN, color='black',linestyle=linestyles[m],label=labels[m])

	# # legend = ax2.legend(loc='upper left')
	# # legend.get_frame().set_facecolor('lightblue')

	ax.set_ylabel(ylabel='GHG from Agriculture (kt CO$_2$e)')
	ax1.set_ylabel( ylabel='Final Exergy (TJ)')
	ax2.set_ylabel(ylabel='GHG from Energy (kt CO$_2$e)',rotation=270)
	ax.set_xlabel(xlabel='Year')
	ax1.set_xlabel(xlabel='Year')
	ax.yaxis.set_label_coords(-0.1,0.5)
	ax2.yaxis.set_label_coords(1.13,0.5)
	ax1.yaxis.set_label_coords(-0.05,0.5)

	if show:
		plt.show()
		if save == True:
			fig.savefig('/Users/bpc/floobits/bpc/Dissertation_Bruno/Figures/data_ghg.pdf', format='pdf', dpi=3600)
		return
	plt.close()
	return

def plotForecast(show,save,data,fold_selector=4):
	MEDIUM_SIZE = 18
	BIGGER_SIZE = 24

	plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	matplotlib.rcParams.update({'errorbar.capsize': 2})
	
	fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=False, 
					            figsize=(15,6))
	ax.grid(linestyle='dotted')
	fig.subplots_adjust(left=0.10, bottom=0.15, right=0.90, top=0.95, wspace=None, hspace=None)

	path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/'
	# path = '/Volumes/PEN/GHG/'
	path1 = path + 'Simple_1_12_2020_logs/'
	path2 = path + 'LSTM_1_12_2020_logs/'
	path3 = path + 'GRU_1_12_2020_logs/'
	# path4 = path + 'conv_2_12_2020_logs/'
	path5 = path + 'conv1D_LSTM_2_12_2020_logs/'
	path6 = path + 'conv1D_GRU_2_12_2020_logs/'

	fold_dates = np.arange(1961,2031,1)
	# paths = [path1, path2, path3, path4, path5, path6]
	paths = [path1, path2, path3, path5, path6]
	error_vec = [228.2/2, 4252.9/2]

	study = '_ghg'
	scenarios = ['avestruz','lince']
	for j,scenario in enumerate(scenarios):
		m_predict = [[]for i in range(len(paths))]
		m_predict_average = [[]for i in range(len(paths))]
		for i, path in enumerate(paths):
			m_predict[i] = deserialization(path,fold_selector,scenario,study,True)
			# print(m_predict[i])
			m_predict_average[i] = np.round(sum(m_predict[i])/len(m_predict[i]),1)
			# print(m_predict_average[i])
			# m_predict_average[i][:, [0,1,2,3,4]] = m_predict_average[i][:, [4,0,1,2,3]]
			# print(m_predict_average[i])
		m_predict_average_total = np.round(sum(m_predict_average)/len(m_predict_average),1)
		print(m_predict_average_total)

		# colors = ['royalblue','darkorange','forestgreen','black','firebrick']
		colors = ['blue','darkgreen']
		linestyles = ["dotted",'dashdot']
		# labels = ['Scenario $\it{P}$','Scenario $\it{O}$']
		# labels = [r'GDP $\searrow$',r'GDP $\nearrow$'])
		labels = [r'Exergy $\searrow$',r'Exergy $\nearrow$']
		ax.plot(fold_dates[49:56],data[49:,4],
										marker="o",
										linestyle="solid",
										color=colors[0],
										linewidth=1.3,alpha=1,markersize=4),
		ax.errorbar(fold_dates[56:56+7],m_predict_average_total[:7,0],
										# yerr=error_vec[0],
										marker="x",
										linestyle=linestyles[j],
										color=colors[0],
										linewidth=1.8,alpha=1,markersize=8),
		ax.plot(fold_dates[56+7-1:],m_predict_average_total[6:,0],
										marker="x",
										linestyle=linestyles[j],
										color=colors[0],
										linewidth=1.8,alpha=1,markersize=8),
		
		if j == 0:
			ax2 = ax.twinx()
		ax2.plot(fold_dates[49:56],data[49:,5],
										marker="o",
										linestyle="solid",
										color=colors[1],
										linewidth=1.3,alpha=1,markersize=4),
		ax2.errorbar(fold_dates[56:56+7],m_predict_average_total[:7,1],
										# yerr=error_vec[1],
										marker="x",
										linestyle=linestyles[j],
										color=colors[1],
										linewidth=1.8,alpha=1,markersize=8),
		ax2.plot(fold_dates[56+7-1:],m_predict_average_total[6:,1],
										marker="x",
										linestyle=linestyles[j],
										color=colors[1],
										linewidth=1.8,alpha=1,markersize=8),

	ax.tick_params(axis='y', labelcolor=colors[0])
	ax2.tick_params(axis='y', labelcolor=colors[1])
	ax.set_xticks(np.arange(2010,2031,2))
	# ax2.set_xticks(np.arange(2017,2031,1))
	# ax.set_yticks(np.arange(5900,6601,60))
	ax.set_yticks(np.arange(5900,6901,80))
	# ax2.set_yticks(np.arange(40000,63101,1400))
	ax2.set_yticks(np.arange(40000,67101,2900))
	ax.set_ylim(5900, 6700.)
	ax2.set_ylim(39000, 67101.)
	ax.ticklabel_format(style='sci',scilimits=(1,2),axis='y')
	ax2.ticklabel_format(style='sci',scilimits=(2,4),axis='y')

	# ax2 = ax.twinx()
	markers = ["o","x"]
	labels2 = ['Historical Values','Forecasted Values']
	for m in range(len(linestyles)):
		ax2.plot(np.NaN, np.NaN, color='black',linestyle=linestyles[m],label=labels[m],markersize=8,linewidth=2.)
	for m in range(len(markers)):
		ax.plot(np.NaN, np.NaN, color='gray',linestyle="None",marker=markers[m],markersize=8,linewidth=2.,
				label=labels2[m])

	ax.legend(bbox_to_anchor=(1.,0), loc='lower right')
	# ax.legend(bbox_to_anchor=(0.49,0.777), loc='lower right')

	legend = ax2.legend(loc='upper left')
	legend.get_frame().set_facecolor('lightblue')

	plt.setp(ax, ylabel=r'GHG from Agriculture (kt CO$_2$e)')
	ax2.set_ylabel(ylabel=r'GHG from Energy (kt CO$_2$e)',rotation=270)
	plt.setp(ax, xlabel='Year')
	ax.yaxis.set_label_coords(-0.092,0.5)
	ax2.yaxis.set_label_coords(1.125,0.5)
	ax.xaxis.set_label_coords(0.5,-0.1)
	# axs[1][0].set_yticks(np.arange(0,36,5))
	# plt.yticks(np.arange(0,36,2))

	# fig.tight_layout()
	# fig.suptitle("Fold 0",y=1.5)
	if show:
		plt.show()
		if save == True:
			fig.savefig('/Users/bpc/floobits/bpc/Dissertation_Bruno/Figures/' + 'ghg' +'.pdf', format='pdf', dpi=3600)
		return
	plt.close()
	return

# fold_selector = 3
# plotTrainingErrors(fold_selector,True,True)
data = loader('/Users/bpc/6º Ano-2º Sem./Tese/Coding/GHG/Dados Finais.csv').to_numpy()
# plotForecast(show=True,save=False,data=data)
# plotData(data,True,False)