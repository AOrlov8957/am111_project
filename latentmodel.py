import util
import csv
import numpy as np
from math import sqrt

class Latentmodel(object):
	'''
	This will be a latent factor model that attempts to utilize standard SVD to
	characterize latent factors for users versus movies
	'''

	def __init__(self):
		pass

	# def fit(self,data,dims):
	# 	# use design matrix rather than dictionary
	# 	self.X = np.array(data)
	# 	self.u,self.s,self.v = np.linalg.svd(self.X,full_matrices=False)
	# 	self.S = np.diag(self.s[:dims])
	# 	# self.p = self.u[:,:dims]
	# 	# self.q = self.v[:dims,:]
	# 	self.p = np.dot(self.u[:,:dims],np.sqrt(self.S))
	# 	self.q = np.dot(np.sqrt(self.S),self.v[:dims,:])

	# 	print self.p.shape
	# 	print self.q.shape

	# 	print np.dot(self.p[3],self.q[:,15])
	# 	pass

	def fit(self,data,dims):
		self.r = np.array(data)
		self.m = self.r.shape[0]
		self.n = self.r.shape[1]
		self.dims = dims
		
		# specify a gradient descent step
		self.gamma_ = 0.1
		self.lambda_ = 0.1

		# initialize p and q
		u,s,v_t = self.lowRankSVD()

		self.p = np.dot(u,np.sqrt(s))
		self.q = np.dot(np.transpose(v_t),np.sqrt(s))
		
		self.fit_mat = np.dot(self.p,np.transpose(self.q))
		self.mat_max = np.max(self.fit_mat)
		self.mat_min = np.min(self.fit_mat)

		pass

		# perform stochastic gradient descent
		# for x in range(5):
		# 	print self.predError()
		# 	for u in range(self.m):
		# 		for i in range(self.n):
		# 			# skip missing values
		# 			if self.r[u][i] == 0.0:
		# 				continue
		# 			e_ui = self.r[u][i] - np.dot(self.p[u],self.q[i])
		# 			if abs(e_ui) > 100:
		# 				self.printStats(u,i)
		# 				return
		# 			print '---'
		# 			print e_ui
		# 			print '---'
		# 			print self.q[i]
		# 			self.q[i] = self.q[i] + self.gamma_*(self.p[u]*e_ui - self.lambda_*self.q[i])
		# 			print self.q[i]
		# 			print '---'
		# 			print self.p[u]
		# 			self.p[u] = self.p[u] + self.gamma_*(self.q[i]*e_ui - self.lambda_*self.p[u])
		# 			print self.p[u]


	def predError(self):
		err = 0.0
		for u in range(self.m):
			for i in range(self.n):
				if self.r[u][i] == 0:
					continue
				err += np.square(self.r[u][i] - np.dot(self.p[u],self.q[i])) + self.lambda_*(np.linalg.norm(self.p[u]) + np.linalg.norm(self.q[i]))

		return err

	def lowRankSVD(self):
		u,s,v = np.linalg.svd(self.r,full_matrices=False)
		s = np.diag(s[:self.dims])
		u = u[:,:self.dims]
		v_T = v[:self.dims,:]

		return u,s,v_T

	def lowRankErr(self):
		U,S,VT = self.lowRankSVD()
		return np.linalg.norm(self.r - np.dot(np.dot(U,S),VT))

	def printStats(self,u,i):
		print "R[u][i] = " + str(self.r[u][i])
		print "Q[i] = " + str(self.q[i])
		print "P[u] = " + str(self.p[u])
		print "e_ui = " + str(self.r[u][i] - np.dot(self.p[u],self.q[i]))

	def predictedRating(self,user,item):
		# return noramlized expected rating
		# return ((self.fit_mat[user][item] - self.mat_min)/(self.mat_max - self.mat_min))*5
		return self.fit_mat[user][item]
				
data = util.design_matrix('ml-100k/u.data')
l = Latentmodel()
l.fit(data,1500)
for item in range(300):
	print l.predictedRating(4,item)
# print l.predError()



