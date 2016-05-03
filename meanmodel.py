import util
import csv
import numpy as np
import sklearn
import scipy as scipy
from math import sqrt

class Meanmodel(object):
	'''
	This will be a simple model as well that will calculate user and movie averages
	to create a simple rating system for any given individual. Here's an example: let's pretend
	that I'm browsing through movies and the average rating of all movies is 3.5 stars. I
	find a movie that's better than average, so maybe it's average rating is 4.0 stars (0.5 stars
	higher than average). I, however, am a curmudgeon that tends to rate movies 0.2 stars below
	the average. This means that my rating for this particular movie would come out to 3.8 stars.
	'''

	def __init__(self):
		pass

	def fit(self,data):
		self.X = data
		# calculate global movie mean
		val = 0.0
		count = 0.0
		for person in self.X.keys():
			for movie in self.X[person]:
				val += self.X[person][movie]
				count += 1

		self.globalMean = val/(count - 1.0)
		pass

	def userMean(self,person):
		# calculate sample mean for a given user
		count = 0.0
		val = 0.0
		for key in self.X[person].keys():
			count += 1.0
			val += self.X[person][key]

		return val/(count)

	def movieMean(self,movie):
		# calculate sample mean for a given movie
		count = 0.0
		val = 0.0
		for person in self.X.keys():
			# get movie average
			if movie not in self.X[person].keys():
				continue
			count += 1.0
			val += self.X[person][movie]

		return val/(count)

	def predictedRating(self,person,movie):
		# get the predicted rating for a single person/movie pair
		return self.globalMean + (self.userMean(person) - self.globalMean) + (self.movieMean(movie) - self.globalMean)

	def getRecommendations(self,person):
		# get predicted rating for a single person over all movies
		u_mean = self.userMean(person)
		# create set of movies
		movie_set = set([])
		for user in self.X.keys():
			for movie in self.X[user]:
				movie_set.add(movie)

		rankings = {}
		max_ = -float('inf')
		min_ = float('inf')
		for movie in movie_set:
			mean = self.movieMean(movie)
			rankings[movie] = self.globalMean + (u_mean - self.globalMean) + (mean - self.globalMean)
			if rankings[movie] > max_:
				max_ = rankings[movie]
			if rankings[movie] < min_:
				min_ = rankings[movie]

		print max_
		print min_

		n_rankings = [(rankings[movie],movie) for movie in rankings.keys()]

		n_rankings.sort()
		n_rankings.reverse()

		return n_rankings

data = util.design_dict('ml-100k/u.data')

m = Meanmodel()
user = 3
movie = 16
m.fit(data)
print m.predictedRating(user,movie)
# print m.getRecommendations(4)






