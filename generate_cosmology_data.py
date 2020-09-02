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
    x = generate_training_parameters(npoints=10)
    y = generate_data(x)
    data = write_data(x, y)
    save_data(data, save_fn)

    save_fn = 'data/cosmology_test.npy'
    x = generate_testing_parameters(ntest=100)
    y = generate_data(x)
    data = write_data(x, y)
    save_data(data, save_fn)


# Generate the parameters that govern the output training set data
def generate_training_parameters(npoints=10):
    nparams = 3
    omegam = np.linspace(0.26, 0.34, npoints)
    sigma8 = np.linspace(0.7, 0.95, npoints)
    omegab = np.linspace(0.038, 0.058, npoints)
    grid = np.meshgrid(omegam, sigma8, omegab)
    # x has shape (nparams, ndata), where ndata = npoints**nparams
    x = np.array([grid[p].flatten() for p in range(nparams)])
    return x

# Generate the parameters that govern the output testing set data
def generate_testing_parameters(ntest=100):
    omegam = random_between(0.26, 0.34, ntest)
    sigma8 = random_between(0.7, 0.95, ntest)
    omegab = random_between(0.038, 0.058, ntest)
    # x has shape (nparams, ndata)
    x = np.array([omegam, sigma8, omegab])
    return x


def random_between(xmin, xmax, n):
    return np.random.rand(n)*(xmax-xmin)+xmin

# Generate the output data that we're interested in emulating
def generate_data(x):
    redshift = 0.0
    rvals = np.linspace(50, 140, 10)

    ndata = x.shape[1]
    y = np.empty((len(rvals), ndata))
    for i in range(ndata):
        print(i)
        om, s8, ob = x[:,i]
        ocdm = om - ob
        m_ncdm = [] #no massive neutrinos
        cosmo = cosmology.Cosmology(Omega0_b=ob, Omega0_cdm=ocdm, m_ncdm=m_ncdm)
        cosmo = cosmo.match(sigma8=s8)
        Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
        CF = cosmology.correlation.CorrelationFunction(Plin)
        y[:,i] = CF(rvals)
    return y


# Write the data as an array of dictionaries
# (because numpy likes to save arrays, not dictionaries)
def write_data(x, y):
    data = []
    ndata = x.shape[1]
    for i in range(ndata):
        data.append({'x': x[:,i], 'y': y[:,i], 'label':i})
    return np.array(data)


# Save the data to a file
def save_data(data, save_fn):
    np.save(save_fn, data)


if __name__=='__main__':
    main()
