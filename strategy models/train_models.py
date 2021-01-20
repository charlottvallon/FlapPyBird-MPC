#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 10:45:36 2021
Script to train GP models for flappy bird
@author: vallon2
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# 1. Pick features for our training set
# inputs: del_x to pipe 1, y of pipe 1, y of pipe 2 (all pipes on screen)
# outputs: y(N)

    # can also attempt: what if we only consider the last pipe?! since we just want to predict the last state. 
    # inputs: del_x to last pipe, y of last pipe
    # outputs: y(N)
    
# we'll need to pick a good N!!!! maybe ours is fine.

X = np.array([0,0,0,0])
y = np.array([0])

cutoff = 30

# 2. load training data from a bunch of saved files. 
list_of_files = glob.glob('data/*.csv') # * means all if need specific format then *.csv
for file in list_of_files:
    training_set = np.loadtxt(open(file,"rb"),delimiter = ",") # 6 x num_samples 
    # [player_y, player_VY, pipe_1_x, pipe_1_y, pipe_2_x, pipe_2_y]
    
    # cut off the last few seconds (where failure occurred)
    training_set = training_set[:,:-cutoff]
    
    # what is the sampling time between these things? one time step. need to be careful when we pick the label!
    X = np.vstack((X, training_set[2:, :-31].T))
    
    # how many actual time steps are we looking ahead? what does N go to? N=24 --> 31! 
    y = np.hstack((y, training_set[0,31:].T))

# get rid of first row of 0s 
y = y[1:]
X = X[1:,:]


# certainly need to standardize these things!!! they should all be the same. could we normalize them all to 0/1? 
# TO DO!!

# split into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=0)



#%% 3. Train the data
t_X_train = torch.from_numpy(X_train).double()
t_y_train = torch.from_numpy(y_train).double()

# save t_X_train and t_y_train so that we can re-load them later after saving the model
# or: can we just pickle this?! what a disaster



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(t_X_train, t_y_train, likelihood)


# Find optimal model hyperparameters
training_iter = 500
model.double()
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(t_X_train)
    # Calc loss and backprop gradients
    loss = -mll(output, t_y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    
#%% Predictions with the model
model.eval()
likelihood.eval()

t_X_test = torch.from_numpy(X_test[100:200,:]).double()
t_y_test = torch.from_numpy(y_test[100:200]).double()

observed_pred = likelihood(model(t_X_test))
lower, upper = observed_pred.confidence_region() # pm 2 standard deviations

with torch.no_grad():
    est_y = observed_pred.mean.numpy()

std = (upper.numpy() - lower.numpy())/2

plt.figure()    
plt.plot(est_y,'r')
plt.plot(est_y + std,'k')
plt.plot(est_y - std, 'k')
plt.plot(y_test[100:200],'b')

#%% to do
# 1. save model
torch.save(model.state_dict(), 'model_state.pth')

# loading (will have to do in flappy_pred): 
# state_dict = torch.load('model_state.pth')
# model = ExactGPModel(train_x, train_y, likelihood)  # Create a new GP model
# model.load_state_dict(state_dict)

# 3. implement current model into the flappy bird thing