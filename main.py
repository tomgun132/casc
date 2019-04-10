import os
import time
import sys
import numpy as np
from model import Model
import data
from scipy import spatial
# import matplotlib.pyplot as plt
from train import Train

# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
num_summs = 5
base_path = "../data/TAC2011/"

top_w = 50
top_k = 4

f = open(base_path + "topics_tiny.txt", "r") # or use topics_tiny.txt for development
for topic in f:
	topic = topic.strip('\n')
	print("compiling...")

	train_proc = Train(base_path, topic)
	num_sents = len(train_proc.lines)
	start = time.time()
	w_summs, Ax = train_proc.train(250, num_summs, 'gru')
	o = open(base_path + "/cascaded/rouge/" + topic, "w") 

	Xx = np.linalg.norm(Ax, axis=0)
	ind = np.argpartition(Xx, -top_k)[-top_k:]
	for k in ind:
		print(train_proc.lines[k])
	print("==================")


	Xk = train_proc.w_concept
	ind = np.argpartition(Xk, -top_k)[-top_k:]
	for k in ind:
		print(train_proc.lines[k])
		#o.write(lines[k])
	print("=================")

	X = Xx
	np.savetxt(base_path + "/cascaded/salience/" + topic + ".atten", X)
	#X = np.linalg.norm(A , axis=0)
	ind = np.argpartition(X, -top_k)[-top_k:]
	for k in ind:
		print(train_proc.lines[k])
		o.write(train_proc.lines[k] + "\n")
	print("=================")
	#######################################

	# top sents
	o_sents = open(base_path + "/cascaded/salience/" + topic + ".sent", "w")
	for i in range(num_sents):
		o_sents.write(str(X[i]) + "\n")
		#print Xi[i] , Xj[i] , X1[i], X2[i], Xk[i]
	o_sents.close()
	print("=================")

	# top words
	o_words = open(base_path + "/cascaded/salience/" + topic + ".word", "w")
	top_w = 20
	Xm = np.sum(w_summs, axis=0)
	ind = np.argsort(-Xm)
	for k in range(top_w):
		print(train_proc.i2w[ind[k]])
	for k in range(len(train_proc.i2w)):
		o_words.write(train_proc.i2w[ind[k]] + " " + str(Xm[ind[k]]) + "\n")
	o_words.close()

	# top words for each top
	o_words = open(base_path + "/cascaded/salience/" + topic + ".tword", "w")
	for i in range(num_summs):
		s_i = w_summs[i,:]
		ind = np.argsort(-s_i)
		for k in range(top_w):
			print(train_proc.i2w[ind[k]] + ", ",)
		print("\n")
		ws = ""
		for k in range(len(train_proc.i2w)):
			ws += train_proc.i2w[ind[k]] + " "
		o_words.write(ws + "\n")
	o_words.close()

	o.close() # end of summary

f.close()
print("Finished.")