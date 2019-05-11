#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:01:11 2019

@author: brgupta
"""

# This is unsupervised  deep learning

# This is for customer froud : for creadit card application dataset

#import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
x = sc.fit_transform(x)

# Training the SOM
# numpy based application for self organizing map : Minisom
# minisom implementation
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x,num_iteration=100)
 
# visualizing the results
from pylab import bone, pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(x):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]],colors[y[i]],markerfacecolor='None',markersize=10, markeredgewidth=2)
    
show()

# Finding the frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)


 
 