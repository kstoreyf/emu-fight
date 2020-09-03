#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:53:26 2020

@author: johannesheyl
"""

import numpy as np
import os
import sklearn
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


class emulator:
    def __init__(self):        
        self.x = []
        self.y = []
        self.number_of_samples = [] # can add any other quantities of interest here
        self.val_loss = []
        
    def read_data(self, file):
        infile = open(file,'rb')
        new_dict = pickle.load(infile)
        infile.close()
        
        for n, i in enumerate(new_dict["input_data"]):
            self.x.append(new_dict["input_data"].values[n][1:])
            
        for data in new_dict["output_data"]:
            self.x.append(new_dict["input_data"].values[n][1:])
                
    def train_random_forest_regressor(self, scale = False, n_estimators = 100):
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.x)
            self.x_scaled = self.scaler.transform(self.x)
            X_train, X_test, y_train, y_test = train_test_split(self.x_scaled, self.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y)
        self.model_rf = RandomForestRegressor(verbose = 0, n_jobs=-1, n_estimators=n_estimators) #n_jobs=-1 parallelises the training
        self.model_rf.fit(X_train, y_train)
        y_pred = self.model_rf.predict(X_test)
        val_loss = mean_squared_error(y_test, y_pred)
        print("Validation Loss: " + str(val_loss))
        self.val_loss_list.append(val_loss)
        
    def train_nn_regressor(self, scale = True, architecture = (512,256,128), activation_func = "tanh"):
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.x)
            self.x_scaled = self.scaler.transform(self.x)
            X_train, X_test, y_train, y_test = train_test_split(self.x_scaled, self.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y)
        self.model_rf = MLPRegressor(hidden_layer_sizes = architecture, activation = activation_func, verbose = True, early_stopping=True)
        self.model_rf.fit(X_train, y_train)
        y_pred = self.model_rf.predict(X_test)
        val_loss = mean_squared_error(y_test, y_pred)
        self.val_loss_list.append(val_loss)
        print("Validation Loss: " + str(self.model_rf.loss_))
        self.number_of_samples.append(len(self.x))
    
    def train_nn_regressor_tf(self, scale = True, architecture = (512,256,128), ndim = 7):
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.x)
            self.x_scaled = self.scaler.transform(self.x)
            X_train, X_test, y_train, y_test = train_test_split(self.x_scaled, self.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y)
        self.model = Sequential()
        self.model.add(Dense(architecture[0], input_dim=ndim, kernel_initializer='normal', activation=tf.nn.leaky_relu))
        for i in architecture[1:]:
            self.model.add(Dense(i, activation=tf.nn.leaky_relu))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        self.model.summary()
        self.history=self.model.fit(np.asarray(self.x), np.asarray(self.y), epochs=30, batch_size=150,  verbose=1, validation_split=0.2)
        self.val_loss_list.append(self.history.history['loss'])
        self.number_of_samples.append(len(self.x))