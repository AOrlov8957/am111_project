'''
Created by Alexander Orlov, 4/19/2016

utils.py is a collection of functions that help with auxiliary things
like data import, cleaning, etc.
'''

import csv
import numpy as np

def design_dict(f_name):
	data = {}
	with open(f_name) as tsvfile:
		# read in tsv
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		max_u = 0
		max_i = 0
		for row in tsvfile:
			s = row.split('\t')
			u_id   = int(s[0])
			i_id   = int(s[1])
			rating = int(s[2])
			ts     = row[3]
			if not u_id in data:
				data[u_id] = {}

			if u_id > max_u:
				max_u = u_id
			if i_id > max_i:
				max_i = i_id

			# create dictionary that links user id and ratings
			data[u_id][i_id] = rating

	return data

def design_matrix(f_name):
	data = design_dict(f_name)

	full_data = [[float('NaN') for i in range(max_i)] for j in range(max_u)]

	# create design matrix
	for k_u in data.keys():
		for k_i in data[k_u].keys():
			full_data[k_u-1][k_i-1] = data[k_u][k_i]

	return full_data