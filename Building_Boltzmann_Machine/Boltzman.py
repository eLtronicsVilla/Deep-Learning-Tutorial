#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:27:06 2019

@author: brgupta
"""

####################################################################################################### 

# 1.The Boltzmann Machine
# 2. Energy based model (EBM)
# 3. Restricted Boltzmann Machine(RBM)
# 4. Contrastive divergence(CD)
# 5. Deep Belief Network(DBN)
# 6. Deep Boltzmann Machine(DBM)

# Artificail Neural Network - Used for regression and classificarion
# Convolution Neural Network - Used for computer vision
# Recurrent Neural Network - Used for time seriese Analysis
# Self organizing Map - Used for feature Detection
# Deep boltzmann Machines - Used for recommendation System
# Auto Encoder - Used for Recommnedation System


# Here in Boltzmann Machine : There are hidden nodes and visible nodes , but there is no output layer
# Boltzmann machines are fundamently different ,they generate information regardless whether it's input node or hidden node
# In visible node they are all node connected between each other.
# You have row of data , just input it ,connections between these visible nodes,There is not adjusting the weights.

# Boltzmann Machine is a representation of certain system. The way these whole model works , It is capable of generating all nodes.
# It's generate Parameter on it's own ,in our case it just generate different case ,system 

#####################################################################################################   
# Restricted boltzmann Machine - restriction is that Hidden node cannot connect to each other and visible node cannot connect to each other.

#----------------------------------------------------------------------------------------------------- 

# Boltzmann Machine :
# 1. In one recommended system customer is going to like the movie ( Yes / No)
# 2. In other recommended system we are going to predict rating ( 1 to 5)

# Importing the library
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat',sep = '::',header=None,engine='python',encoding ='latin-1')
users = pd.read_csv('ml-1m/users.dat',sep = '::',header=None,engine='python',encoding ='latin-1')
ratings = pd.read_csv('ml-1m/movies.dat',sep = '::',header=None,engine='python',encoding ='latin-1')


# Prepare the training set and test set
training_set=pd.read_csv('ml-100k/u1.test',delimiter = '\t')
training_set = np.array(training_set, dtype ='int')
test_set=pd.read_csv('ml-100k/u1.base',delimiter = '\t')
test_set = np.array(test_set, dtype ='int')

# getting the number of user and movie
# total number of users and total number of movies
nb_users = max(max(training_set[:,0]),max(test_set[:,0]))
nb_movies = max(max(training_set[:,1]),max(test_set[:,1]))

# converting the data into array with user in lines and movie in column
def convert(data):
    new_data = []
    for id_users in range(1,nb_users + 1):
        id_movies = data[:,1][data[:,0]==id_users]
        id_ratings = data[:,1][data[:,0]==id_users]
        ratings =np.zeros(nb_movies)
        ratings[id_movies -1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# convert the data into Torch tensors 

# tensors are simply array that contains single data type , instead of being numpy array this is pytorch array
# numpy array will be less efficient to create a model, then we are using tensors - pytorch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# converting the rating into the binary rating1 like and 0 not like
training_set [training_set == 0] = -1
training_set [training_set == 1] = 0
training_set [training_set == 2] = 0
training_set [training_set >= 3] = 1

test_set [test_set == 0] = -1
test_set [test_set == 1] = 0
test_set [test_set == 2] = 0
test_set [test_set >= 3] = 1

# Creating the architecture of neural network
# we are going to build probabilistic graphical model

class RBM():
    def __init__(self,nv,nh):
        # self corresponds to the object created for this class later, nv - number of visible node
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)

# sample edge function is nothing but sigmoid function
    def sample_h(self,x): # thisis for hidden neurons
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        # here it's given the visible node
        p_h_given_v = torch.sigmoid(activation)     # this is the sigmoid to this activation function
        return p_h_given_v, torch.bernoulli(p_h_given_v)  # gives the rating of the movie. 0 - correspons=ds the hidden network correspods the sampling 
     
        # this is for visible neuron
    def sample_v(self,y): # thisis for hidden neurons
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        # here it's given the visible node
        p_v_given_h = torch.sigmoid(activation)     # this is the sigmoid to this activation function
        return p_v_given_h, torch.bernoulli(p_v_given_h)  # gives the rating of the movie. 0 - correspons=ds the hidden network correspods the sampling 
    def train(self,v0,vk,ph0,phk):
         self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
         self.b += torch.sum((v0-vk),0)
         self.a += torch.sum((ph0-phk),0)
         
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv,nh)
            
# Training the RBM - to update weight and bias
nb_epoch = 10
for epoch in range(1,nb_epoch+1):
    # we have RMS , we will measure the difference between predicted rating and real rating 0 and 1
    # we have couple of option for measuring the loss , RBM .We need to introduce the loss variable 
    train_loss = 0
    s = 0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk=training_set[id_user:id_user+batch_size]
        v0 =training_set[id_user:id_user+batch_size]
        ph0,_= rbm.sample_h(v0)
        for k in range (10):
            # we have to do sample edging : call sample_h
            _,hk = rbm.sample_h(vk) # second sample of hk node become input to the next vk
            _,vk = rbm.sample_v(hk) # return the second sample of visible node
            vk[v0<0] = v0[v0<0]
            # compute PHK
            phk,_ = rbm.sample_h(vk)
            rbm.train(v0,vk,ph0,phk)
            train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
            # update the counter 
            s += 1.
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))


# Testing RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v=training_set[id_user:id_user+1] # here training set is the inout to activate the hidden neuron
    vt =test_set[id_user:id_user+1] # it contains the original rating to the test set 
    # to get our prediction of the test set rating , do we need to apply again to k step contrasted diversion
    # we need to make one step: Its not the random walk ,it's principle of the blind walk, 100 steps to straight line
    if len(vt[vt>=0])>0:
        _,h = rbm.sample_h(v) # second sample of hk node become input to the next vk
        _,v = rbm.sample_v(h) # return the second sample of visible node
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        # update the counter 
        s += 1.
print('test loss: '+str(test_loss/s))
