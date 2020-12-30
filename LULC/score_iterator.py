import glob, os
import json
from json import JSONEncoder
import collections
import numpy as np

# custom JSON encoder to serialise ndarray
class NumpyArrayEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return JSONEncoder.default(self, obj)

# read lines looking for score in dictionary
def read_past_scores(filename):
	file = open(filename, "r")
	lines = file.readlines()
	file.close()
	for line in lines:
		dic = json.loads(line)
		if 'score' in dic:
			return (dic['score']) if (dic['score']) != None else -1

def get_scores(path,nr_best_trials):
	filepaths = {}
	# search within models optimisation data for all folds available (not sorted)
	for filepath in sorted(glob.iglob(path+'/*', recursive=True)):
		filepaths['filepath'+(filepath[-1])]=filepath

	# save all scores and respective filepaths in a list of dictionaries
	scores = [[] for x in range(len(filepaths))]
	for j,key in enumerate(filepaths):
		# print(filepaths[key]+'/*/**.json')
		for filepath in glob.iglob(filepaths[key]+'/*/**.json', recursive=True):
			scores[j].append({'score': read_past_scores(filepath), 'filepath': filepath})

	# sort dictionary entries by score key
	filenames = [[]for i in range(nr_best_trials)]
	for j in range(nr_best_trials):
		for score in scores: # score is a dictionary {'score' , 'filepath'}
			score.sort(key = lambda i: i['score'])
			filenames[j].append(score[j]['filepath']) # lowest score
			# print(filenames[j])
			print(score[j]['score']) # j_th lowest score (mse) in a given fold

	# look up hyperparameters in paths with lowest score, saving them
	hyperparameters = [[]for i in range(nr_best_trials)]
	for i in range(nr_best_trials):
		for filename in filenames[i]:
			file = open(filename, "r")
			lines = file.readlines()
			file.close()
			print(filename)
			for line in lines:
				dic = json.loads(line)
				if 'hyperparameters' in dic:
					hyperparameters[i].append(dic['hyperparameters']['values'])
					# print(hyperparameters[i])
			dic = []
	return hyperparameters

# directory = 'Simple_15_11_2020'
# path = '/Users/bpc/6º Ano-2º Sem./Tese/Coding/LULC/' + directory
# hp = get_scores(path,3)
# with open('/Volumes/PEN/' + directory +'_logs/' + 'architecture' + '.json', 'w', encoding='utf-8') as f:
# # with open(path +'_logs/' + 'architecture' + '.json', 'w', encoding='utf-8') as f:
# 		json.dump(hp, f, ensure_ascii=False, indent=4, cls=NumpyArrayEncoder)
# # print(hp[2])
# print(hp[0][2]) # hyperparameters for model with lowest score in a given fold (last parameter)
# print(hp[1][2]) # hyperparameters for model with 2nd lowest score in 1st fold
# print(hp[2][2]) # hyperparameters for model with 3rd lowest score in 1st fold



