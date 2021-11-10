# !/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import random
import time
import math
import os
import shutil


  
import copy    # array-copying convenience
import sys     # max float

# -----------------------------------

# Neuroevolution of FNN using 1. Real Coded Genetic Alg (G3PCX) and 2. PSO for classification problems

import torch
import torch.nn as nn

torch.manual_seed(2)
random.seed(2)
np.random.seed(2)

class MLP(nn.Module):
    def __init__(self, ip, hid, num_classes = 10):
        super(MLP, self).__init__()
        self.input_size = ip
        self.output_size = num_classes
        
        self.feature = nn.Sequential(
            nn.Linear(ip, hid),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hid, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        x=x.type(torch.FloatTensor)
        feat = x = self.feature(x)
        x = self.classifier(x)

        if return_feat:
            return x, feat
        else:
            return x

    def fit(self , lr, error):
      optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)
      error.backward()
      optim.step()
      # print('yo')
      self.zero_grad()

def sort_pop(list1, list2):
  idx = np.argsort(np.array(list2))[::-1]
  z=list(np.array(list1)[idx])

  return z

class transfer_bagging():  
	def __init__(self, Ninput, Nhidden, Noutput, n, max_epochs, traindata, testdata):
		super(transfer_bagging, self).__init__()
		self.n = n

		self.input_size = Ninput
		self.hidden_size = Nhidden
		self.output_size = Noutput

		self.max_epochs=max_epochs

		self.criterion = nn.CrossEntropyLoss()

		self.traindata = traindata
		self.bootdata=[]
		for i in range(self.n):
		  self.bootdata.append(self.traindata[np.random.choice(range(self.traindata.shape[0]), size=int((2*self.traindata.shape[0])/self.n)),:])
		self.testdata = testdata

	def error(self, model, data):
	  y = data[:, self.input_size]
	  y=y.type(torch.LongTensor)  
	  x = data[:, 0:self.input_size]
	  out = model(x)
	  return self.criterion(out, y)

	def accuracy(self, model, data):
	  y = data[:, self.input_size]
	  y=y.type(torch.LongTensor)  
	  x = data[:, 0:self.input_size]
	  out = model(x)
	  pred = torch.argmax(out, axis=1)
	  correct = (pred == y).sum()
	  acc = correct / y.shape[0]
	  # print(acc)
	  # iitr
	  return acc

	def accuracy_avg(self, swarm, data):
	  y = data[:, self.input_size]
	  y=y.type(torch.LongTensor)  
	  x = data[:, 0:self.input_size]
	  s = torch.zeros((data.shape[0],self.output_size))
	  for i in range(self.n):
	    s += swarm[i](x)
	  pred = torch.argmax(s, axis=1)
	  correct = (pred == y).sum()
	  acc = correct / y.shape[0]
	  # print(acc)
	  # iitr
	  return acc

	def create_bags(self):


		rnd = random.Random(0)
		# create n random particles 

		swarm = [MLP(self.input_size, self.hidden_size, self.output_size) for i in range(self.n)]
		swarm_error = [sys.float_info.max] * self.n

		epoch = 0

		for i in range(self.n):
		  swarm_error[i] = self.error(swarm[i], torch.tensor(self.bootdata[i]))

		while epoch < self.max_epochs:
			for gen in range(10):
				for i in range(self.n): # process each particle
					# print(self.n)
					# print(i)
					swarm_error[i] = self.error(swarm[i], torch.tensor(self.bootdata[i]))
					swarm[i].fit(1e-3, swarm_error[i])
					swarm_error[i] = self.error(swarm[i], torch.tensor(self.bootdata[i]))

			swarm = sort_pop(swarm, swarm_error)
			swarm_error.sort()


			if epoch % 10 == 0 or epoch == 0:
				print("Epoch = " + str(epoch) + " best error = %.7f" % swarm_error[0])
				test_error = self.error(swarm[0], torch.tensor(self.testdata))
				loss_train = swarm_error[0]
				loss_test = test_error

				print(loss_train ,  'classification_perf RMSE train * ' )   
				print(loss_test , 'classification_perf  RMSE test * ' )
				

			for i in range(self.n):
			  if i<self.n//2:
			    self.bootdata[i] = self.traindata[np.random.choice(range(self.traindata.shape[0]), size=int((self.traindata.shape[0])/self.n)),:]
			  else:
			    swarm[i].load_state_dict(swarm[i-self.n//2].state_dict())

			epoch += 1
		for i in range(self.n): # process each particle

			print(self.accuracy(swarm[i],torch.tensor(self.testdata)))
		print('avg',self.accuracy_avg(swarm,torch.tensor(self.testdata))) # process each particle

		iitr

		# train_per, rmse_train = self.swarm_classification_perf(best_swarm_pos, 'train')
		# test_per, rmse_test = self.swarm_classification_perf(best_swarm_pos, 'test')

		return loss_train, loss_test



 



def main():


	#problem = 8

	method = 'pso'    # or 'rcga'

	for problem in range(5, 9) : 


		separate_flag = False # dont change 


		if problem == 1: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-red.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-red"
			hidden = 50
			ip = 11 #input
			output = 10 
		if problem == 3: #IRIS
			data  = np.genfromtxt('DATA/iris.csv',delimiter=';')
			classes = data[:,4].reshape(data.shape[0],1)-1
			features = data[:,0:4]

			separate_flag = True
			name = "iris"
			hidden = 8  #12
			ip = 4 #input
			output = 3 
			#NumSample = 50000
		if problem == 2: #Wine Quality White
			data  = np.genfromtxt('DATA/winequality-white.csv',delimiter=';')
			data = data[1:,:] #remove Labels
			classes = data[:,11].reshape(data.shape[0],1)
			features = data[:,0:11]
			separate_flag = True
			name = "winequality-white"
			hidden = 50
			ip = 11 #input
			output = 10 
			#NumSample = 50000
		if problem == 4: #Ionosphere
			traindata = np.genfromtxt('DATA/Ions/Ions/ftrain.csv',delimiter=',')[:,:-1]
			testdata = np.genfromtxt('DATA/Ions/Ions/ftest.csv',delimiter=',')[:,:-1]
			name = "Ionosphere"
			hidden = 15 #50
			ip = 34 #input
			output = 2 

			#NumSample = 50000
		if problem == 5: #Cancer
			traindata = np.genfromtxt('DATA/Cancer/ftrain.txt',delimiter=' ')[:,:-1]
			testdata = np.genfromtxt('DATA/Cancer/ftest.txt',delimiter=' ')[:,:-1]
			name = "Cancer"
			hidden = 8 # 12
			ip = 9 #input
			output = 2 
			#NumSample =  50000

			# print(' cancer')

		if problem == 6: #Bank additional
			data = np.genfromtxt('DATA/Bank/bank-processed.csv',delimiter=';')
			classes = data[:,20].reshape(data.shape[0],1)
			features = data[:,0:20]
			separate_flag = True
			name = "bank-additional"
			hidden = 50
			ip = 20 #input
			output = 2 
			#NumSample = 50000
		if problem == 7: #PenDigit
			traindata = np.genfromtxt('DATA/PenDigit/train.csv',delimiter=',')
			testdata = np.genfromtxt('DATA/PenDigit/test.csv',delimiter=',')
			name = "PenDigit"
			for k in range(16):
				mean_train = np.mean(traindata[:,k])
				dev_train = np.std(traindata[:,k])
				traindata[:,k] = (traindata[:,k]-mean_train)/dev_train
				mean_test = np.mean(testdata[:,k])
				dev_test = np.std(testdata[:,k])
				testdata[:,k] = (testdata[:,k]-mean_test)/dev_test
			ip = 16
			hidden = 30
			output = 10 

			#NumSample = 50000
		if problem == 8: #Chess
			data  = np.genfromtxt('DATA/chess.csv',delimiter=';')
			classes = data[:,6].reshape(data.shape[0],1)
			features = data[:,0:6]
			separate_flag = True
			name = "chess"
			hidden = 25
			ip = 6 #input
			output = 18 


		
		#Separating data to train and test
		if separate_flag is True:
			#Normalizing Data
			for k in range(ip):
				mean = np.mean(features[:,k])
				dev = np.std(features[:,k])
				features[:,k] = (features[:,k]-mean)/dev
			train_ratio = 0.6 #Choosable
			indices = np.random.permutation(features.shape[0])
			traindata = np.hstack([features[indices[:np.int(train_ratio*features.shape[0])],:],classes[indices[:np.int(train_ratio*features.shape[0])],:]])
			testdata = np.hstack([features[indices[np.int(train_ratio*features.shape[0])]:,:],classes[indices[np.int(train_ratio*features.shape[0])]:,:]])



		topology = [ip, hidden, output]


		outfile_ga=open('result.txt','a+')
		outfile_pso=open('result_.txt','a+')


		for run in range(1, 2) :  

			max_epochs = 100
			pop_size =  50

			timer = time.time()

			bagger  =  transfer_bagging(ip, hidden, output, pop_size, max_epochs, traindata, testdata)

			rmse_train, rmse_test = bagger.create_bags()

			print(rmse_train,  'classification_perf RMSE train * pso' )   
			print(rmse_test, 'classification_perf  RMSE test * pso' )

			timer2 = time.time()
			timetotal = (timer2 - timer) /60


			allres =  np.asarray([ problem, run, train_per, test_per, rmse_train, rmse_test, timetotal]) 
			np.savetxt(outfile_pso,  allres   , fmt='%1.4f', newline='   '  )
			np.savetxt(outfile_pso,  ['  PSO'], fmt="%s", newline=' \n '  )




   
   



	 


 














if __name__ == "__main__": main()
