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
import matplotlib.pyplot as plt

class emulator:
    def __init__(self):        
        self.x = []
        self.y = []
        self.number_of_samples = [] # can add any other quantities of interest here
        self.val_loss = []
        
    def read_data(self, training_file, test_file):
        infile_train = open(training_file,'rb')
        new_dict_train = pickle.load(infile_train)
        infile_train.close()
        
        input_train = new_dict_train['input_data']
        output_train = new_dict_train['output_data']

        self.xs_train = np.array(input_train.drop(columns=['object_id']))
        self.ys_train = np.array(output_train.drop(columns=['object_id']))
        self.extra_train = new_dict_train['extra_input']
        self.r_vals = self.extra_train['r_vals']
        
        infile_test = open(test_file,'rb')
        new_dict_test = pickle.load(infile_test)
        infile_test.close()
        
        input_test = new_dict_test['input_data']
        output_test = new_dict_test['output_data']

        self.xs_test = np.array(input_test.drop(columns=['object_id']))
        self.ys_test = np.array(output_test.drop(columns=['object_id']))    
        
        self.n_params = input_train.shape[1]-1
        self.n_values = output_train.shape[1]-1
        self.number_test = input_test.shape[0]


                            
    def train_random_forest_regressor(self, x, y, scale = False):
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(x)
            self.x_scaled = self.scaler.transform(x)
            X_train, X_test, y_train, y_test = train_test_split(self.x_scaled, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y)
        self.model_rf = RandomForestRegressor(verbose = 0, n_jobs=-1, n_estimators=100) #n_jobs=-1 parallelises the training
        self.model_rf.fit(X_train, y_train)
        y_pred = self.model_rf.predict(X_test)
        
    def train_nn_regressor(self, scale = True, architecture = (512,256,128), activation_func = "tanh"):
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.x)
            self.x_scaled = self.scaler.transform(self.x)
            X_train, X_test, y_train, y_test = train_test_split(self.x_scaled, self.y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.x, self.y)
        self.model_nn = MLPRegressor(hidden_layer_sizes = architecture, activation = activation_func, verbose = True, early_stopping=True)
        self.model_nn.fit(X_train, y_train)
        y_pred = self.model_nn.predict(X_test)
        val_loss = mean_squared_error(y_test, y_pred)
        self.val_loss_list.append(val_loss)
        print("Validation Loss: " + str(self.model_nn.loss_))
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
    
 
    def plot_data(self, regressor):
        regrs = np.empty(self.n_values, dtype=object)
        for j in range(self.n_values):
            ys_train_r = self.ys_train[:,j]
            ys_test_r = self.ys_test[:,j]
            if regressor == "RF":
                self.train_random_forest_regressor(x = self.xs_train, y = ys_train_r.ravel())
                self.model_rf.score(self.xs_test, ys_test_r)
            regrs[j] = self.model_rf        
                
        ys_predict = np.zeros((self.number_test, self.n_values))
        for j in range(self.n_values):  
            ys_predict_r = regrs[j].predict(self.xs_test)
            ys_predict[:,j] = ys_predict_r
            
        n_plot = int(0.2*self.number_test)
        idxs = np.random.choice(np.arange(self.number_test), n_plot)
        color_idx = np.linspace(0, 1, n_plot)
        colors = np.array([plt.cm.rainbow(c) for c in color_idx])
        
        plt.figure(figsize=(8,6))
        for i in range(n_plot):
            ys_test_plot = self.ys_test[idxs,:][i]
            ys_predict_plot = ys_predict[idxs][i]
            if i==0:
                label_test = 'truth'
                label_predict = 'emu_prediction'
            else:
                label_test = None
                label_predict = None
            plt.plot(self.r_vals[:self.n_values], ys_test_plot, alpha=0.8, label=label_test, marker='o', markerfacecolor='None', ls='None', color=colors[i])
            plt.plot(self.r_vals[:self.n_values], ys_predict_plot, alpha=0.8, label=label_predict, color=colors[i])
        plt.xlabel('$r$')
        plt.ylabel(r'$\xi(r)$')
        plt.legend()
        
        
# sample use
#x = emulator()
#x.read_data("/Users/johannesheyl/Downloads/cosmology_train_big.pickle", "/Users/johannesheyl/Downloads/cosmology_test.pickle")
#x.plot_data(regressor="RF")

