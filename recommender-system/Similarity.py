from mrjob.job import MRJob
from mrjob.step import MRStep
import sys
from math import sqrt

class MRSimilarity(MRJob):

	def mapper(self, _, line):
		u, i, r_ui = line.split()
		yield int(u), (int(i), float(r_ui))

	def reducer_1(self, key, value):
		u, items_score = key, list(value)
		ru_hat = sum([x[1] for x in items_score]) / len(items_score)
		for idx, v1 in enumerate(items_score):
			i, r_ui = v1
			for v2 in items_score[idx:]:
				j, r_uj = v2
				if i < j:
					yield '{}_{}'.format(i, j), (r_ui, r_uj, ru_hat, u)
				else:
					yield '{}_{}'.format(j, i), (r_uj, r_ui, ru_hat, u)

	def reducer_2(self, key, value):
		(i, j), value = map(int, key.split('_')), list(value) 
		prod = []
		norm_i = []
		norm_j = []
		for v in value:
			r_ui, r_uj, ru_hat, u = v
			prod.append((r_ui - ru_hat) * (r_uj - ru_hat))
			norm_i.append((r_ui - ru_hat) ** 2)
			norm_j.append((r_uj - ru_hat) ** 2)
	
		norm = sqrt(sum(norm_i)) * sqrt(sum(norm_j))
		if norm == 0:
			sim_ij = 0.0
		else:
			sim_ij = sum(prod) / norm
		sim_ij = sim_ij if sim_ij >= 0 else 0.0 
		if i == j:
			yield i,  (j, sim_ij)
		else:
			yield i,  (j, sim_ij)
			yield j, (i, sim_ij)

	def steps(self):
		return [MRStep(mapper=self.mapper, reducer=self.reducer_1),
				MRStep(reducer=self.reducer_2)]



