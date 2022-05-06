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
c = 340 # velocity of sound in air (in m/s)
fs = 16000 # in Hertz

f_model = open(THETA_PATH+'Theta_P={}-nfft={}.pkl'.format(P, nfft), 'wb')
P_pos = np.random.rand(P,3)
mic_pos = np.array([[6.5, 4.49,1.5], [6.5, 4.51,1.5]])
Dilatation_taille_salle = np.array([8,9,3])

P_pos_PT = np.einsum("pt,t ->pt", P_pos, Dilatation_taille_salle)
P_pos_PTM = np.einsum("pt,m -> ptm", P_pos_PT, np.ones(2))
r_PTM = P_pos_PTM - np.einsum("mt,p->ptm",mic_pos,np.ones(P))
Dist_PM = np.einsum("ptm -> pm",r_PTM**2)
Theta_FPM = np.empty((nfft,P,M))
for f in range(nfft):
    Theta_FPM[f] = 1/Dist_PM*np.exp(-1j*Dist_PM*f*fs/2/c)

if save:
    pkl.dump(Theta_FPM, f_model)
    f_model.close()



