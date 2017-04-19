import numpy as np
import pandas as pd
import argparse

from cross_validation import mean_squared_error
from Similarity import MRSimilarity

def find_similarity(train_csv, N):
	mr_job = MRSimilarity(args=[train_csv])
	similarity = np.zeros((N + 1, N + 1))
	with mr_job.make_runner() as runner:
		runner.run()
		for line in runner.stream_output():
			i, value = mr_job.parse_output_line(line)
			j, sim_ij = value
			similarity[int(i), int(j)] = float(sim_ij)  	
	return similarity

def find_ratings(data_train, data_test, s):
	answer = np.zeros_like(data_test.Rating.values)
	norm = np.sum(s, axis=1)
	norm[norm == 0.0] = 1.0
	predict = np.dot(data_train, s) / norm
	for i, (uid, mid) in enumerate(data_test[['UserID', 'MovieID']].values):
		ans = predict[uid, mid]
		if ans == 0:
			answer[i] = np.sum(data_train[uid, :]) / (np.count_nonzero(data_train[uid, :]) or 1)
			
		else:
			answer[i] = ans
	return answer 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='neighborhood')
	parser.add_argument('train_csv')
	parser.add_argument('test_csv')
	parser.add_argument('distance_matrix')
	args = parser.parse_args()

	data_train = pd.read_csv(args.train_csv, sep=' ', header=None)
	data_train.columns = ['UserID', 'MovieID', 'Rating']
	data_test = pd.read_csv(args.test_csv, sep=' ', header=None)
	data_test.columns = ['UserID', 'MovieID', 'Rating']
	Nmovies = data_train.MovieID.max()
	Nusers = data_train.UserID.max()

	s_m = find_similarity(args.train_csv, Nmovies)
	np.save(args.distance_matrix, s_m)
	#s_m = np.load('distance_matrix.npy')

	user_data = np.zeros((Nusers + 1, Nmovies + 1))
	for value in data_train.values:
		u, i, r_ui = value
		user_data[int(u), int(i)] = r_ui

	NList = [1, 5, 10, 20, 50, 100, 1000, 5000]
	for N in NList:
		new_s_m = np.zeros_like((s_m))
		sort_idxs = s_m.argsort(axis=0)[::-1]
		for i in range(new_s_m.shape[1]):
			new_idx = sort_idxs[:N, i]
			new_s_m[new_idx, i] = s_m[new_idx, i]
		
		answer_train = find_ratings(user_data, data_train, new_s_m)
		print ('MSE(Train):{}, N:{}'.format(mean_squared_error(data_train['Rating'].values, answer_train, scale=1.0), N))

		answer_test = find_ratings(user_data, data_test, new_s_m)
		print ('MSE(Test):{}, N:{}'.format(mean_squared_error(data_test['Rating'].values, answer_test, scale=1.0), N))

		print ('--------------------------------------------------------------------')
