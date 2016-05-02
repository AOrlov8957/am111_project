import util
import csv
import numpy as np
import sklearn
from sklearn.cross_validation import train_test_split
import scipy as sp
from math import sqrt

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
		print self.sim_pearson(1,1)
		print self.sim_distance(1,1)
				
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

		sum_of_squares = np.sum([pow(float(self.X[p1][item]) - float(self.X[p2][item]),2) for item in self.X[p1] if item in self.X[p2]])

		return 1.0/(1.0+sum_of_squares)
		# return sum_of_squares

	def vec_sim_pearson(self,p1,p2):
		# return vectorized pearson correlation coefficient
		x,y = self.vectorize(self.X,p1,p2)
		return sp.stats.pearsonr(x,y)[0]

	def sim_pearson(self,p1,p2):
		# get mutually rated items
		si = {}
		for item in self.X[p1]:
			if item in self.X[p2]:
				si[item] = 1

		# number of elements in common
		n = len(si)

		# if no ratings in common, return 0
		if n == 0: return 0

		# Add up all the preferences
		sum1=sum([self.X[p1][it] for it in si])
		sum2=sum([self.X[p2][it] for it in si])

		# Sum up the squares
		sum1Sq=sum([pow(self.X[p1][it],2) for it in si])
		sum2Sq=sum([pow(self.X[p2][it],2) for it in si])

		# Sum up the products
		pSum=sum([self.X[p1][it]*self.X[p2][it] for it in si])

		# Calculate Pearson score
		num=pSum-(sum1*sum2/n)
		den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))

		if den==0: return 0

		r=num/den

		return r

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

	def topMatches(self,person,n=5,similarity=sim_pearson):
		scores=[(similarity(self,person,other),other)
		for other in self.X if other!=person]

		# Sort the list so the highest scores appear at the top
		scores.sort()
		scores.reverse()
		return scores[0:n]

	def getRecommendations(self,person,similarity):
		totals={}
		simSums={}
		for other in self.X:
			# don't compare me to myself
			if other==person:
				continue
			sim=similarity(person,other)

			# ignore scores of zero or lower
			if sim<=0:
				continue

			for item in self.X[other]:
				# only score movies I haven't seen yet
				if item not in self.X[person] or self.X[person][item]==0:
					# Similarity * Score
					totals.setdefault(item,0)
					totals[item]+=self.X[other][item]*sim
					# Sum of similarities
					simSums.setdefault(item,0)
					simSums[item]+=sim

		# Create the normalized list
		rankings=[(total/simSums[item],item) for item,total in totals.items()]

		# Return the sorted list
		rankings.sort()
		rankings.reverse()
		return rankings


data = util.design_dict('ml-100k/u.data')

m = Baseline()
m.fit(data)
print m.getRecommendations(4,similarity=m.sim_pearson)

