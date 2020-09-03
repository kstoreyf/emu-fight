#!/usr/bin/env python3
"""
Created on Tue Sep  1 2020

@author: kstoreyf
"""
import numpy as np
import nbodykit
import pandas as pd
import pickle

from nbodykit import cosmology


def main():
    save_fn = '../data/cosmology_train.pickle'
    x = generate_training_parameters(n_train=1000)
    y, extra_input = generate_data(x)
    input_data, output_data = format_data(x, y, 
                                          objs_id=None)
    data_to_save = make_data_to_save(input_data, output_data, 
                                     extra_input)
    save_data(data_to_save, save_fn)

    save_fn = '../data/cosmology_test.pickle'
    x = generate_testing_parameters(n_test=100)
    y, extra_input = generate_data(x)
    input_data, output_data = format_data(x, y, 
                                          objs_id=None)
    data_to_save = make_data_to_save(input_data, output_data, 
                                     extra_input)
    save_data(data_to_save, save_fn)


# Generate the parameters that govern the output training set data
def generate_training_parameters(n_train=1000):
    n_params = 3
    n_points = n_train**(1./float(n_params))
    assert abs(round(n_points) - n_points) < 1e-12, f"n_train must be a power of {n_params} because we're making a high-dimensional grid." 
    n_points = round(n_points)
    omega_m = np.linspace(0.26, 0.34, n_points)
    sigma_8 = np.linspace(0.7, 0.95, n_points)
    omega_b = np.linspace(0.038, 0.058, n_points)
    grid = np.meshgrid(omega_m, sigma_8, omega_b)
    # x has shape (n_params, n_train), where n_train = n_points**n_params
    x = np.array([grid[p].flatten() for p in range(n_params)])
    return x


# Generate the parameters that govern the output testing set data
def generate_testing_parameters(n_test=100):
    omega_m = random_between(0.26, 0.34, n_test)
    sigma_8 = random_between(0.7, 0.95, n_test)
    omega_b = random_between(0.038, 0.058, n_test)
    # x has shape (n_params, n_test)
    x = np.array([omega_m, sigma_8, omega_b])
    return x


def random_between(xmin, xmax, n):
    return np.random.rand(n)*(xmax-xmin)+xmin


# Generate the output data that we're interested in emulating
def generate_data(x):
    redshift = 0.0
    r_vals = np.linspace(50, 140, 10)
    extra_input = {'redshift': redshift, 'r_vals': r_vals}

    n_data = x.shape[1]
    y = np.empty((len(r_vals), n_data))
    for i in range(n_data):
        print(i)
        om, s8, ob = x[:,i]
        ocdm = om - ob
        m_ncdm = [] #no massive neutrinos
        cosmo = cosmology.Cosmology(Omega0_b=ob, Omega0_cdm=ocdm, m_ncdm=m_ncdm)
        cosmo = cosmo.match(sigma8=s8)
        plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
        cf = cosmology.correlation.CorrelationFunction(plin)
        y[:,i] = cf(r_vals)
    return y, extra_input


# Format data into pandas data frames
def format_data(x_input, y_output, objs_id=None):
    number_objs = len(x_input[0,:])
    number_outputs = len(y_output[:,0])
    if objs_id is None:
        objs_id = [f'obj_{i}'for i in np.arange(number_objs)]
    
    input_data = pd.DataFrame()
    input_data['object_id'] = objs_id
    input_data[r'$\Omega_m$'] = x_input[0,:]
    input_data[r'$\sigma_8$'] = x_input[1,:]
    input_data[r'$\Omega_b$'] = x_input[2,:]
    
    output_data = pd.DataFrame()
    output_data['object_id'] = objs_id
    for i in np.arange(number_outputs):
        output_data[r'$\xi(r_{})$'.format(i)] = y_output[i, :]
    return input_data, output_data


# Format the data to save it
def make_data_to_save(input_data, output_data, extra_input=None):
    data_to_save = {'input_data': input_data, 
                    'output_data': output_data}
    if extra_input is not None:
        data_to_save['extra_input'] = extra_input
    return data_to_save


# Save the data to a file
def save_data(data, save_fn):
    with open(save_fn, 'wb') as f:
        pickle.dump(data, f, protocol=3)


if __name__=='__main__':
    main()
