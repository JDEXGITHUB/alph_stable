# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:54:33 2022

@author: MSI
"""

import h5py
import numpy as np
import pickle as pkl

save = True
THETA_PATH = "./data/"


# number of diracs, number of frequency bins, number of mics
P = 72
nfft = 1024
M = 2


f_model = open(THETA_PATH+'Theta_P={}-nfft={}.pkl'.format(P, nfft), 'wb')
Theta_FPM = np.random.rand(nfft,P,M)

if save:
    pkl.dump(Theta_FPM, f_model)
    f_model.close()



