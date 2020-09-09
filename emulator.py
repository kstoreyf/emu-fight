#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:53:26 2020

@author: johannesheyl
"""
import matplotlib
import numpy as np
import pickle
import sklearn
import tensorflow as tf

from collections import defaultdict
from matplotlib import pylab
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

matplotlib.rcParams['figure.dpi'] = 80
params = {'figure.figsize': (7, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large'}
pylab.rcParams.update(params)


class emulator:
    def __init__(self):
        self.x = []
        self.y = []
        # can add any other quantities of interest `number_of_samples`
        self.number_of_samples = []
        self.val_loss = []
        self.models = defaultdict(dict)

    def read_data(self, training_file, test_file, scale=False,
                  normalize_y=False):
        self.scale = scale
        self.normalize_y = normalize_y

        infile_train = open(training_file, 'rb')
        new_dict_train = pickle.load(infile_train)
        infile_train.close()

        input_train = new_dict_train['input_data']
        output_train = new_dict_train['output_data']

        self.xs_train_orig = np.array(input_train.drop(columns=['object_id']))
        self.ys_train_orig = np.array(output_train.drop(columns=['object_id']))
        self.extra_train = new_dict_train['extra_input']
        self.r_vals = self.extra_train['r_vals']

        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.xs_train_orig)
            self.xs_train = self.scaler.transform(self.xs_train_orig)
        else:
            self.xs_train = self.xs_train_orig

        if normalize_y:
            self.y_mean = np.mean(self.ys_train_orig, axis=0)
            self.ys_train = self.ys_train_orig/self.y_mean
        else:
            self.ys_train = self.ys_train_orig

        infile_test = open(test_file, 'rb')
        new_dict_test = pickle.load(infile_test)
        infile_test.close()

        input_test = new_dict_test['input_data']
        output_test = new_dict_test['output_data']

        self.xs_test_orig = np.array(input_test.drop(columns=['object_id']))
        self.ys_test_orig = np.array(output_test.drop(columns=['object_id']))
        if scale:
            self.xs_test = self.scaler.transform(self.xs_test_orig)
        else:
            self.xs_test = self.xs_test_orig

        if normalize_y:
            self.ys_test = self.ys_test_orig/self.y_mean
        else:
            self.ys_test = self.ys_test_orig

        self.n_params = input_train.shape[1]-1
        self.n_values = output_train.shape[1]-1
        self.number_test = input_test.shape[0]

    def train(self, regressor_name, **kwargs):
        self.models[regressor_name]['regressors'] = np.empty(self.n_values,
                                                             dtype=object)
        train_func = None
        if regressor_name == "RF":
            train_func = self.train_random_forest_regressor
        if regressor_name == "ANN":
            train_func = self.train_ann_regressor
        if regressor_name == "SVM":
            train_func = self.train_svm_regressor
        if regressor_name == "DTree":
            train_func = self.train_decision_tree_regressor

        for j in range(self.n_values):
            ys_train_r = self.ys_train[:, j]
            model = train_func(x=self.xs_train, y=ys_train_r.ravel(), **kwargs)
            self.models[regressor_name]['regressors'][j] = model

    def test(self, regressor_name, metric):
        error_message = f"{regressor_name} not yet trained!"
        assert regressor_name in self.models, error_message
        metric_funcs = {"r2": sklearn.metrics.r2_score,
                        "mse": sklearn.metrics.mean_squared_error}
        error_message = ('{} not recognized! options are: {}'
                         ''.format(metric, metric_funcs.keys()))
        assert metric in metric_funcs, error_message
        scores = np.empty(self.n_values)
        for j in range(self.n_values):
            ys_predict = self.models[regressor_name]['ys_predict']
            scores[j] = metric_funcs[metric](self.ys_test_orig[:, j],
                                             ys_predict[:, j])
        self.models[regressor_name][metric] = scores

    # assumes model has a predict method that takes x values
    def predict_test_set(self, regressor_name):
        ys_predict = np.zeros((self.number_test, self.n_values))
        for j in range(self.n_values):
            model = self.models[regressor_name]['regressors'][j]
            ys_predict[:, j] = model.predict(self.xs_test)
        if self.normalize_y:
            ys_predict = ys_predict*self.y_mean
        self.models[regressor_name]['ys_predict'] = ys_predict

    def train_ann_regressor(self, x, y, hidden_layer_sizes=(14,),
                            alpha=0.00028, activation='relu', solver='lbfgs',
                            tol=1e-6, learning_rate_init=0.001, beta_1=0.9,
                            beta_2=0.985):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             alpha=alpha, activation=activation,
                             random_state=1, max_iter=10000, solver=solver,
                             tol=tol, learning_rate_init=learning_rate_init,
                             beta_1=beta_1, beta_2=beta_2).fit(x, y)
        return model

    def train_svm_regressor(self, x, y, kernel='rbf', epsilon=5e-4,
                            C=11, gamma=0.09, tol=1e-6):
        model = SVR(kernel=kernel, epsilon=epsilon, C=C, gamma=gamma,
                    tol=tol).fit(x, y)
        return model

    def train_decision_tree_regressor(self, x, y):
        model = DecisionTreeRegressor(random_state=0,
                                      criterion="mae").fit(x, y)
        return model

    def train_random_forest_regressor(self, x, y, n_estimators=100,
                                      n_jobs=None):
        # n_jobs=-1 parallelises the training
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs)
        model.fit(x, y)
        return model

    def train_nn_regressor_tf(self, x, y, architecture=(512, 256, 128),
                              ndim=7):
        model = Sequential()
        model.add(Dense(architecture[0], input_dim=ndim,
                  kernel_initializer='normal',
                  activation=tf.nn.leaky_relu))
        for i in architecture[1:]:
            model.add(Dense(i, activation=tf.nn.leaky_relu))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam',
                           metrics=['mse', 'mae'])
        model.summary()
        history = model.fit(np.asarray(x), np.asarray(y),
                            epochs=30, batch_size=150,  verbose=1,
                            validation_split=0.2)
        self.val_loss_list.append(history.history['loss'])
        self.number_of_samples.append(len(self.x))
        return model

    def plot_predictions(self, regressor_name, frac=0.2):
        n_plot = int(frac*self.number_test)
        np.random.seed(42)
        idxs = np.random.choice(np.arange(self.number_test), n_plot)
        color_idx = np.linspace(0, 1, n_plot)
        colors = np.array([plt.cm.rainbow(c) for c in color_idx])

        for i in range(n_plot):
            ys_test_plot = self.ys_test_orig[idxs, :][i]
            model = self.models[regressor_name]
            ys_predict_plot = model['ys_predict'][idxs, :][i]
            if i == 0:
                label_test = 'truth'
                label_predict = 'emu_prediction'
            else:
                label_test = None
                label_predict = None
            plt.plot(self.r_vals, ys_test_plot, alpha=0.8,
                     label=label_test, marker='o', markerfacecolor='None',
                     ls='None', color=colors[i])
            plt.plot(self.r_vals, ys_predict_plot, alpha=0.8,
                     label=label_predict, color=colors[i])
        plt.xlabel('$r$')
        plt.ylabel(r'$\xi(r)$')
        plt.legend()

    def plot_training(self):
        plt.plot(self.r_vals, self.ys_train_orig.T, alpha=0.8, lw=0.5)
        plt.xlabel('$r$')
        plt.ylabel(r'$\xi(r)$')

    def plot_accuracy(self, metric, regressor_names=None):
        if regressor_names is None:
            regressor_names = self.models.keys()
        plt.figure()
        for rn in regressor_names:
            error_message = ('must first run emu.test(regressor_name, metric) '
                             'for regressor_name {} and metric {}!'
                             ''.format(rn, metric))
            assert metric in self.models[rn].keys(), error_message
            scores = self.models[rn][metric]
            plt.plot(self.r_vals, scores, label=rn)
        plt.xlabel('$r$')
        plt.ylabel(metric)
        plt.legend()

# sample use
'''
x = emulator()
x.read_data("/Users/johannesheyl/Downloads/cosmology_train_big.pickle",
            "/Users/johannesheyl/Downloads/cosmology_test.pickle")
x.plot_data(regressor="RF")
'''
