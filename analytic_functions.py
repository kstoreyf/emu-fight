#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:54:32 2020

@author: johannesheyl
"""

import numpy as np

def eggbox_function(x, y):
    return np.exp((2+np.cos(x/2)*np.cos(y/2))**5)

def rosenbruck_function(x):
    N = np.asarray(x).shape[0]
    
    output = []
    for i in range(N):
        output.append((1-x[i]**2) + 100*(x[i+1]-x[i]**2)**2)
        
    return np.sum(output)