import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import mean_squared_error
from data_pipeline import loader, lag

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
def deserialization(path,fold_selector,scenario,apply_criteria=False):
	with open(path +'fold_' + str(fold_selector) + '_' + scenario + '_lulc_gdp' + '.json', "r") as r:
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

def deserialization_not_scenario(path,fold_selector,apply_criteria=False):
	with open(path +'fold_' + str(fold_selector) + '.json', "r") as r:
		dics = json.load(r)
		cont = dics['Context']
		stats = dics['Statistics']
		y_test_extracted_inverted = np.asarray(dics['Targets']['Real Values'])
		y_preds_extracted_inverted = np.asarray(dics['Targets']['Predicted Values'])
		rmses = stats['Final_RMSE']
		if apply_criteria:
			all_rmse = []
			sum_rmse = []
			for rmse in rmses.values():
				all_rmse.append(rmse)
			for i in range(len(all_rmse)):
				sum_rmse.append(np.round(np.sum(all_rmse[i]),1))
			indexes = [i for i,v in enumerate(sum_rmse) if v > 15]
			for index in sorted(indexes, reverse=True):
				y_preds_extracted_inverted = np.delete(y_preds_extracted_inverted,index,axis=0)
				del rmses['RMSE'+str(index)]
		return y_test_extracted_inverted, y_preds_extracted_inverted, rmses

def swap_columns(your_list):
	n_col = len(your_list)
	for item in your_list:
		# exit(item)
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
	SMALL_SIZE = 8
	MEDIUM_SIZE = 12
	BIGGER_SIZE = 16

	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
	plt.rc('legend', fontsize=10)    # legend fontsize
	plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
	# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/'
	path = '/Volumes/PEN/LULC/'
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
		m_true[i], m_predict[i], rmse = deserialization_not_scenario(path,fold_selector,True)
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


	fig, axs = plt.subplots(ncols=2, nrows=9, sharex=False, sharey=False, 
				            figsize=(8,11))
	fig.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.99, wspace=0.15, hspace=0.2)
    
	#1st graph (1,1) (row,col)
	#2nd graph (2,1)
	#3rd graph (3,1)
	#4th graph (1,2)
	#5th graph (2,2)
	#6th graph (3,2)

	counter = 0
	colors = ['royalblue','darkorange','forestgreen','black','firebrick']
	for j in range(2):
		for k in range(9):
			for i in range(len(m_true[0])):
				if (k == 0 or k == 1 or k == 2) and j == 0:
					counter = 0
				elif (k == 3 or k == 4 or k == 5) and j == 0:
					counter = 1
				elif (k == 6 or k == 7 or k == 8) and j == 0:
					counter = 2
				elif (k == 0 or k == 1 or k == 2) and j == 1:
					counter = 3
				elif (k == 3 or k == 4 or k == 5) and j == 1:
					counter = 4
				elif (k == 6 or k == 7 or k == 8) and j == 1:
					counter = 5	
				else:
					exit("Not good")

				if k == 0 or k == 3 or k == 6:
					label = "M"+str(counter+1)
				else:
					label = None

				axs[k][j].plot(fold_dates[fold_selector],m_true[0][:,i],marker="o",
														linestyle="solid",
														# color='rgbkm'[i],
														color=colors[i],
														linewidth=0.6,alpha=1,markersize=2)
														#label="M1")
				axs[k][j].plot(fold_dates[fold_selector],m_predict_average[counter][:,i],
														marker="x",
														linestyle="dashdot",
														# color='rgbkm'[i],
														color=colors[i],
														linewidth=0.6,alpha=0.7,markersize=5)
														#label="SM"+str(i))
			
	# zoom-in / limit the view to different portions of the data
	d = .015  # how big to make the diagonal lines in axes coordinates
	for l in range(2):
		if fold_selector == 0:
			axs[0][l].set_ylim(35, 37)  # outliers only
			axs[1][l].set_ylim(15., 25.5) # most of the data
			axs[2][l].set_ylim(4., 5.)  # most of the data
			axs[3][l].set_ylim(35, 37)  # outliers only
			axs[4][l].set_ylim(14., 25.8) # most of the data
			axs[5][l].set_ylim(4., 5.)  # most of the data
			axs[5][l].set_yticks(np.arange(4,6,1))
			axs[6][l].set_ylim(35, 37)  # outliers only
			axs[7][l].set_ylim(15., 25.) # most of the data
			axs[8][l].set_ylim(4., 5.)  # most of the data
			axs[8][l].set_yticks(np.arange(4,6,1))

		if fold_selector == 1:
			axs[0][l].set_ylim(35, 37)  # outliers only
			axs[1][l].set_ylim(17., 23.) # most of the data
			axs[1][l].set_yticks(np.arange(17,24,2))
			axs[2][l].set_ylim(4., 5.)  # most of the data
			axs[2][l].set_yticks(np.arange(4,6,1))
			axs[3][l].set_ylim(34, 37)  # outliers only
			axs[3][l].set_yticks(np.arange(34,38,1))
			axs[4][l].set_ylim(17., 23.) # most of the data
			axs[4][l].set_yticks(np.arange(17,24,2))
			axs[5][l].set_ylim(4., 5.)  # most of the data
			axs[5][l].set_yticks(np.arange(4,6,1))
			axs[6][l].set_ylim(34, 36)  # outliers only
			axs[6][l].set_yticks(np.arange(34,37,1))
			axs[7][l].set_ylim(18., 24.) # most of the data
			axs[7][l].set_yticks(np.arange(18,25,2))
			axs[8][l].set_ylim(4., 5.)  # most of the data
			axs[8][l].set_yticks(np.arange(4,6,1))
		if fold_selector == 2:
			axs[0][l].set_ylim(35, 37)  # outliers only
			axs[1][l].set_ylim(18., 22.1) # most of the data
			axs[1][l].set_yticks(np.arange(18,23,2))
			axs[2][l].set_ylim(4., 5.1)  # most of the data
			axs[2][l].set_yticks(np.arange(4,6,1))
			axs[3][l].set_ylim(35, 37)  # outliers only
			axs[3][l].set_yticks(np.arange(35,38,1))
			axs[4][l].set_ylim(17., 23.) # most of the data
			axs[4][l].set_yticks(np.arange(17,24,2))
			axs[5][l].set_ylim(4., 5.1)  # most of the data
			axs[5][l].set_yticks(np.arange(4,6,1))
			axs[6][l].set_ylim(35, 37)  # outliers only
			axs[6][l].set_yticks(np.arange(35,38,1))
			axs[7][l].set_ylim(18., 22.) # most of the data
			axs[7][l].set_yticks(np.arange(18,23,2))
			axs[8][l].set_ylim(4., 5.1)  # most of the data
			axs[8][l].set_yticks(np.arange(4,6,1))
		if fold_selector == 3:
			axs[0][l].set_ylim(35, 37)  # outliers only
			axs[1][l].set_ylim(18., 22.1) # most of the data
			axs[2][l].set_ylim(4, 5.1)  # most of the data
			axs[3][l].set_ylim(35, 37)  # outliers only
			axs[4][l].set_ylim(18., 22.1) # most of the data
			axs[5][l].set_ylim(4, 5.1)  # most of the data
			axs[6][l].set_ylim(35, 37)  # outliers only
			axs[7][l].set_ylim(18., 21.) # most of the data
			axs[7][l].set_yticks(np.arange(18,22,1))
			axs[8][l].set_ylim(4, 5.1)  # most of the data

		if l == 1:
			axs[0][l].set_yticklabels([])
			axs[1][l].set_yticklabels([])
			axs[2][l].set_yticklabels([])
			axs[3][l].set_yticklabels([])
			axs[4][l].set_yticklabels([])
			axs[5][l].set_yticklabels([])
			axs[6][l].set_yticklabels([])
			axs[7][l].set_yticklabels([])
			axs[8][l].set_yticklabels([])


		for m in range(0,7,3):
			# axs[0+m][l].xaxis.tick_top()
			axs[0+m][l].spines['bottom'].set_visible(False)
			axs[0+m][l].tick_params(labelbottom=False)  # don't put tick labels at the top
			axs[0+m][l].xaxis.set_ticks_position('none')
			axs[1+m][l].spines['top'].set_visible(False)
			axs[1+m][l].spines['bottom'].set_visible(False)
			axs[1+m][l].xaxis.set_ticks_position('none')
			axs[1+m][l].tick_params(labeltop=False)  # don't put tick labels at the top
			axs[1+m][l].tick_params(labelbottom=False)  # don't put tick labels at the top
			axs[2+m][l].tick_params(labeltop=False)  # don't put tick labels at the top
			if m != 6:
				axs[2+m][l].tick_params(labelbottom=False)  # don't put tick labels at the top
			axs[2+m][l].spines['top'].set_visible(False)
			axs[2+m][l].xaxis.tick_bottom()

			# arguments to pass to plot, just so we don't keep repeating them
			kwargs = dict(transform=axs[0+m][l].transAxes, color='k', clip_on=False)
			axs[0+m][l].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
			axs[0+m][l].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

			kwargs = dict(transform=axs[1+m][l].transAxes, color='k', clip_on=False)
			axs[1+m][l].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
			axs[1+m][l].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

			kwargs.update(transform=axs[1+m][l].transAxes)  # switch to the bottom axes
			axs[1+m][l].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
			axs[1+m][l].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

			kwargs.update(transform=axs[2+m][l].transAxes)  # switch to the bottom axes
			axs[2+m][l].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
			axs[2+m][l].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

	legend1 = axs[0][0].legend(['M1  '],loc='upper left', shadow=False, markerscale=0, handlelength=0)
	legend2 = axs[3][0].legend(['M2  '],loc='upper left', shadow=False, markerscale=0, handlelength=0)
	legend3 = axs[6][0].legend(['M3  '],loc='upper left', shadow=False, markerscale=0, handlelength=0)
	legend4 = axs[0][1].legend(['M4  '],loc='upper left', shadow=False, markerscale=0, handlelength=0)
	legend5 = axs[3][1].legend(['M5  '],loc='upper left', shadow=False, markerscale=0, handlelength=0)
	legend6 = axs[6][1].legend(['M6  '],loc='upper left', shadow=False, markerscale=0, handlelength=0)
	legends = [legend1, legend2, legend3, legend4, legend5, legend6]
	for legend in legends:
		legend.get_frame().set_facecolor('lightblue')
		


	categories = [ 'Perm. Pasture', 'Perm. Culture & Arable Land', 'Forest', 'Urban/Artificial', 'Other']
	styles = ["solid", "dashdot"]
	markers = ["o","x"]
	labels = ['True Values', 'Predicted Values']

	for m in range(len(styles)):
		axs[8][0].plot(np.NaN, np.NaN, color='gray', linestyle=styles[m],
			marker=markers[m], markersize=5, label=labels[m])
	for i in range(len(m_true[0])):
		axs[8][1].plot(np.NaN, np.NaN, color=colors[i], label=categories[i])

	axs[8][0].legend(bbox_to_anchor=(0.78, -1.18), loc='lower right')
	axs[8][1].legend(bbox_to_anchor=(0.09, -1.9), loc='lower left')

	plt.setp(axs[4][0], ylabel='Land Use Distribution (%)')
	plt.setp(axs[8][0], xlabel='Year')
	axs[4][0].yaxis.set_label_coords(-0.18, 0.5)
	axs[8][0].xaxis.set_label_coords(-0.04, -0.3)

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
	data[:, [0,1,2,3,4,5,6]] = data[:, [6,0,1,2,3,4,5]]
	# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=False, 
	# 					figsize=(8,11))
	###############################################################################
	fig = plt.figure(figsize=(15,8))
	fig.subplots_adjust(left=0.06, bottom=0.33, right=0.99, top=0.975, wspace=0.25, hspace=None)
	gs = fig.add_gridspec(2, 2)
	
	axs = []
	# spans two rows:
	axs.append(fig.add_subplot(gs[:, 0]))
	
	axs.append(fig.add_subplot(gs[0, 1]))
	axs.append(fig.add_subplot(gs[1, 1]))
	axs[1].get_shared_x_axes().join(axs[2])
	##############################################################################
	
	fold_dates = np.arange(1961,2017,1)
	# print(fold_dates)
	# print(len(fold_dates))
	# print(data[:,0:5])
	# exit()

	categories = [ 'Perm. Pasture', 'Perm. Culture & Arable Land', 'Forest', 'Urban/Artificial', 'Other']
	covariates = ['GDP', 'Exergy']
	colors = ['royalblue','darkorange','forestgreen','black','firebrick']
	for i in range(5):
		axs[0].plot(fold_dates,np.transpose(np.array(data[:,i])),marker="o",
												linestyle="solid",
												# color='rgbkm'[i],
												color=colors[i],
												linewidth=1.25,
												alpha=1,
												markersize=3,
												label=categories[i])#,
												#label="M1")
		# axs[0].title.set_text("M"+str(1))
	axs[1].plot(fold_dates,np.array(data[:,5]),
												marker="o",
												linestyle="solid",
												# color='rgbkm'[i],
												color='blue',
												linewidth=1.25,
												alpha=0.7,
												markersize=3),
												#label="SM"+str(i))
	axs[2].plot(fold_dates,np.array(data[:,6]),
												marker="o",
												linestyle="solid",
												# color='rgbkm'[i],
												color='blue',
												linewidth=1.25,
												alpha=0.7,
												markersize=3),
												#label="SM"+str(i))
	axs[0].grid(linestyle='dotted')
	axs[1].grid(linestyle='dotted')
	axs[2].grid(linestyle='dotted')
	plt.setp(axs[0], ylabel='Land Use Distribution (%)')
	plt.setp(axs[1], ylabel='GDP@2015 ref. levels\n (Mrd EURO-PTE)')
	plt.setp(axs[2], ylabel='Final Exergy (TJ)')
	axs[2].ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
	plt.setp(axs[0], xlabel='Year')
	axs[0].xaxis.set_label_coords(0.01, -0.06)
	axs[0].yaxis.set_label_coords(-0.09, 0.5)
	axs[1].yaxis.set_label_coords(-0.1, 0.5)
	axs[2].yaxis.set_label_coords(-0.1, 0.5)
	axs[0].set_xticks(np.arange(1961,2017,6))
	axs[1].set_xticks(np.arange(1961,2017,6))
	axs[1].tick_params(
				    axis='x',          # changes apply to the x-axis
				    which='major',      # both major and minor ticks are affected
				    bottom=True,      # ticks along the bottom edge are off
				    top=False,         # ticks along the top edge are off
				    labelbottom=False) # labels along the bottom edge are off
	axs[2].set_xticks(np.arange(1961,2017,6))
	axs[0].set_yticks(np.arange(0,40,3))
	axs[1].set_yticks(np.arange(0,201,25))
	axs[2].set_yticks(np.arange(0,34000,4000))
	axs[0].legend(bbox_to_anchor=(0.13, -0.5), loc='lower left')

	if show:
		plt.show()
		if save == True:
			fig.savefig('/Users/bpc/floobits/bpc/Dissertation_Bruno/Figures/data.pdf', format='pdf', dpi=3600)
		return
	plt.close()
	return

def plotForecast(show,save,scenario,data,fold_selector=4):
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

	data[:, [0,1,2,3,4,5,6]] = data[:, [6,0,1,2,3,4,5]]

	# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/'
	path = '/Volumes/PEN/LULC/'
	path1 = path + 'Simple_15_11_2020_logs/'
	path2 = path + 'LSTM_15_11_2020_logs/'
	path3 = path + 'GRU_15_11_2020_logs/'
	path4 = path + 'conv_15_11_2020_logs/'
	path5 = path + 'conv1D_LSTM_15_11_2020_logs/'
	path6 = path + 'conv1D_GRU_15_11_2020_logs/'

	# fold_dates = np.arange(2017,2032,1)
	fold_dates = np.arange(1961,2032,1)
	paths = [path1, path2, path3, path4, path5, path6]
	error_vec = [1.4/2, 1.7/2, 0.5/2, 0.4/2, 1.2/2]

	m_predict = [[]for i in range(len(paths))]
	m_predict_average = [[]for i in range(len(paths))]
	for i, path in enumerate(paths):
		m_predict[i] = deserialization(path,fold_selector,scenario,True)
		# print(m_predict[i])
		m_predict_average[i] = np.round(sum(m_predict[i])/len(m_predict[i]),1)
		# print(m_predict_average[i])
		m_predict_average[i][:, [0,1,2,3,4]] = m_predict_average[i][:, [4,0,1,2,3]]
		# print(m_predict_average[i])
	m_predict_average_total = np.round(sum(m_predict_average)/len(m_predict_average),1)
	print(m_predict_average_total)

	fig, (ax,ax3,ax4) = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=False, 
				            figsize=(14,7))


	fig.subplots_adjust(left=0.075, bottom=0.4, right=0.987, top=0.98, wspace=None, hspace=None)

	##################
	# categories = [ 'Perm. Pasture', 'Perm. Culture & Arable Land', 'Forest', 'Urban/Artificial', 'Other']
	# covariates = ['GDP', 'Exergy']
	# colors = ['royalblue','darkorange','forestgreen','black','firebrick']

	# print(m_predict_average_total[8:-1,0])
	# exit(fold_dates[56+7+1:-1])
	counter = 0
	colors = ['royalblue','darkorange','forestgreen','black','firebrick']
	for i in range(len(m_predict_average_total[0])):
		# historical data
		ax.plot(fold_dates[49:56],np.transpose(np.array(data[49:,i])),
													marker="o",
													linestyle="solid",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.3,alpha=1,markersize=4),
													#label="SM"+str(i))
		# forecasted data
		ax.errorbar(fold_dates[56:56+7],m_predict_average_total[:7,i],
													# yerr=error_vec[i],
													marker="x",
													linestyle="dashdot",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.8,alpha=1,markersize=8),
													#label="SM"+str(i))
		# forecasted data
		ax.plot(fold_dates[56+7-1:-1],m_predict_average_total[6:-1,i],
													marker="x",
													linestyle="dashdot",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.8,alpha=1,markersize=8),
													#label="SM"+str(i))
													# historical data
		ax3.plot(fold_dates[49:56],np.transpose(np.array(data[49:,i])),
													marker="o",
													linestyle="solid",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.3,alpha=1,markersize=4),
													#label="SM"+str(i))
		ax3.errorbar(fold_dates[56:56+7],m_predict_average_total[:7,i],
													# yerr=error_vec[i],
													marker="x",
													linestyle="dashdot",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.8,alpha=1,markersize=8),
													#label="SM"+str(i))
		ax3.plot(fold_dates[56+7-1:-1],m_predict_average_total[6:-1,i],
													marker="x",
													linestyle="dashdot",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.8,alpha=1,markersize=8),
													#label="SM"+str(i))
													# historical data
		ax4.plot(fold_dates[49:56],np.transpose(np.array(data[49:,i])),
													marker="o",
													linestyle="solid",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.3,alpha=1,markersize=4),
													#label="SM"+str(i))
		ax4.errorbar(fold_dates[56:56+7],m_predict_average_total[:7,i],
													# yerr=error_vec[i],
													marker="x",
													linestyle="dashdot",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.8,alpha=1,markersize=8),
													#label="SM"+str(i))
		ax4.plot(fold_dates[56+7-1:-1],m_predict_average_total[6:-1,i],
													marker="x",
													linestyle="dashdot",
													# color='rgbkm'[i],
													color=colors[i],
													linewidth=1.8,alpha=1,markersize=8),
													#label="SM"+str(i))

	categories = [ 'Perm. Pasture', 'Perm. Culture & Arable Land', 'Forest', 'Urban/Artificial', 'Other']
	styles = ["solid","dashdot"]
	markers = ["o","x"]
	labels = ['Historical Values','Forecasted Values']
	counter = 0
	# for i in range(len(m_predict_average_total[0])):
	# 	ax.plot(np.NaN, np.NaN, color=colors[i], label=categories[i],linewidth=2.0)
	

	# for m in range(len(styles)):
	# 	ax4.plot(np.NaN, np.NaN, color='gray',linestyle=styles[m],marker=markers[m],
	# 		markersize=8,linewidth=2.,label=labels[m])

	# ax.legend(bbox_to_anchor=(0.88,-4.75), loc='lower right')
	# ax4.legend(bbox_to_anchor=(0.17,-1.45), loc='lower left')

	plt.setp(ax3, ylabel='Land Use Distribution (%)')
	plt.setp(ax4, xlabel='Year')
	ax3.yaxis.set_label_coords(-0.05,0.48)
	ax4.xaxis.set_label_coords(0.01,-0.35)
	if scenario == 'lince':
		# legend = ax3.legend([r'Scenario $\it{O}$ (no COVID-19)   '],
		legend = ax3.legend([r'GDP $\nearrow$   '],
						loc='upper left',	
						shadow=False,
						markerscale=0,
						handlelength=0,
						bbox_to_anchor=(0,2.2),
						fontsize=15)
	elif scenario == 'avestruz':
		# legend = ax3.legend([r'Scenario $\it{P}$ (no COVID-19)   '],)
		legend = ax3.legend([r'GDP $\searrow$   '],
						loc='upper left',	
						shadow=False,
						markerscale=0,
						handlelength=0,
						bbox_to_anchor=(0,2.2),
						fontsize=15)
	else:
		print("Scenario unavailable")
		return
	legend.get_frame().set_facecolor('lightblue')


	ax.set_yticks(np.arange(35,38,1))
	ax3.set_yticks(np.arange(17,24,1))
	ax4.set_yticks(np.arange(4,7,1))
	ax4.set_xticks(np.arange(2010,2031,2))
	ax.xaxis.set_ticks_position('none')
	ax3.xaxis.set_ticks_position('none')

	# zoom-in / limit the view to different portions of the data
	ax.set_ylim(35, 37)  # outliers only
	ax3.set_ylim(18.5, 21.8) # most of the data
	ax4.set_ylim(4, 6)  # most of the data

	# hide the spines between ax and ax2
	ax4.spines['top'].set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.spines['bottom'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax4.xaxis.tick_top()
	ax3.tick_params(labeltop=False)  # don't put tick labels at the top
	ax3.tick_params(labelbottom=False)  # don't put tick labels at the top
	ax.tick_params(labeltop=False)  # don't put tick labels at the top
	ax4.xaxis.tick_bottom()

	d = .007  # how big to make the diagonal lines in axes coordinates
	# arguments to pass to plot, just so we don't keep repeating them
	kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
	ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

	kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
	ax3.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

	kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
	ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
	ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

	kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
	ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
	ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


	# axs[1][0].set_yticks(np.arange(0,36,5))
	# plt.yticks(np.arange(0,36,2))

	# fig.tight_layout()
	# fig.suptitle("Fold 0",y=1.5)
	if show:
		plt.show()
		if save == True:
			fig.savefig('/Users/bpc/floobits/bpc/Dissertation_Bruno/Figures/' + scenario + '_lulc_gdp' +'.pdf', format='pdf', dpi=3600)
		return
	plt.close()
	return
 
# fold_selector = 3
# plotTrainingErrors(fold_selector,True,True)
data = loader('/Users/bpc/6º Ano-2º Sem./Tese/Uso do Solo/Dados/Dados Finais.csv').to_numpy()
plotForecast(show=True,save=False,scenario='lince',data=data)
# plotData(data,True,True)