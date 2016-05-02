import util
import csv
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
import scipy as sp

class Baseline(object):
	'''
	This baseline model is heavily based on the initial CF example from
	programming collective intelligence. I followed the example carefully because
	I think it's a good method to use for a baseline, and though there's very
	little new code here, It gives us a good starting point. I used the same concepts,
	however I'm going to calculated distance measures in a different way. Rather
	than using summations of individual elements I'll vectorize the inputs to speed
	up calculations and stay truer to the later model styles we'll be using.
	'''

	# initialize 
	def __init__(self):
		pass

	# calculate summary statistics and fit baseline model
	def fit(self, data):
		self.X = data
		print self.sim_pearson(1,4)
		print self.sim_distance(1,4)
				
		pass

	def vec_sim_distance(self,p1,p2):
		# return vectorized euclidean distance
		x,y = self.vectorize(self.X,p1,p2)
		dist = np.linalg.norm(x-y,2)
		if dist == 0:
			return 0
		else:
			return 1/dist

	def sim_distance(self,p1,p2):
		# return non-vectorized euclidean distance
		si = {}
		for item in self.X[p1]:
			if item in self.X[p2]:
				si[item] = 1

		# if no rating, return 0
		if len(si) == 0:
			return 0

		sum_of_squares = np.sum([pow(self.X[p1][item] - self.X[p2][item],2) for item in self.X[p1] if item in self.X[p2]])

		return 1/(1+sum_of_squares)

	def vec_sim_pearson(self,p1,p2):
		# return vectorized pearson correlation coefficient
		x,y = self.vectorize(self.X,p1,p2)
		return sp.stats.pearsonr(x,y)[0]

	def sim_pearson(self,p1,p2):
		# get mutually rated items
		

	def topMatches(self,person,n=5,similarity=sim_distance):
		# calculate scores of all other reviewers
		scores=[(similarity(self,person,other),other) for other in self.X if other != person]
		# sort list in place
		scores.sort()
		scores.reverse()
		return scores[0:n]

	def vectorize(self,data,p1,p2):
		# get list of shared ratings
		si = {}
		for item in data[p1]:
			if item in data[p2]:
				si[item]=1
		# transform to vectors
		x = []
		y = []
		for item in si.keys():
			x.append(data[p1][item])
			y.append(data[p2][item])
		# return as np arrays
		return np.array(x),np.array(y)


data = util.design_dict('ml-100k/u.data')

m = Baseline()
m.fit(data)
print m.topMatches(4,n=1000)