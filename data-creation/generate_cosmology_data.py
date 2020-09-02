#!/usr/bin/env python3
"""
Created on Tue Sep  1 2020

@author: kstoreyf
"""
import numpy as np
import nbodykit
from nbodykit import cosmology


def main():
    save_fn = 'data/cosmology_train.npy'
    x = generate_training_parameters(n_train=1000)
    y = generate_data(x)
    data = write_data(x, y)
    save_data(data, save_fn)

    save_fn = 'data/cosmology_test.npy'
    x = generate_testing_parameters(n_test=100)
    y = generate_data(x)
    data = write_data(x, y)
    save_data(data, save_fn)


# Generate the parameters that govern the output training set data
def generate_training_parameters(n_train=1000):
    n_params = 3
    n_points = n_train**(1./float(n_params))
    assert abs(round(n_points) - n_points) < 1e-12, f"n_train must be a power of {n_params}! (because we're making a high-dimensional grid" 
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

    n_data = x.shape[1]
    y = np.empty((len(r_vals), n_data))
    for i in range(n_data):
        print(i)
        om, s8, ob = x[:,i]
        ocdm = om - ob
        m_ncdm = [] #no massive neutrinos
        cosmo = cosmology.Cosmology(Omega0_b=ob, Omega0_cdm=ocdm, m_ncdm=m_ncdm)
        cosmo = cosmo.match(sigma8=s8)
        Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
        CF = cosmology.correlation.CorrelationFunction(Plin)
        y[:,i] = CF(r_vals)
    return y


# Write the data as an array of dictionaries
# (because numpy likes to save arrays, not dictionaries)
def write_data(x, y):
    data = []
    n_data = x.shape[1]
    for i in range(n_data):
        data.append({'x': x[:,i], 'y': y[:,i], 'id':i})
    return np.array(data)


# Save the data to a file
def save_data(data, save_fn):
    np.save(save_fn, data)


if __name__=='__main__':
    main()
