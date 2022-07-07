# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:54:33 2022

@author: MSI
"""

import numpy as np
import pickle as pkl

save = True
THETA_PATH = "./data/"

# number of diracs, number of frequency bins, number of mics
P = 72
nfft = 512
M = 2
c = 340 # velocity of sound in air (in m/s)
fs = 44100 # in Hertz
N = 2   #number of sources
oracle = True


nx, ny = (4,9)   # 6*6=P//2
x = np.linspace(1.5, 3, nx)
y = np.linspace(2.5, 6.5, ny)
xv, yv = np.meshgrid(x, y)

P_pos_PT = np.array([xv,yv,1.5 * np.ones((ny,nx))])  #1.5m de hauteur sont positionnés les instruments
P_pos_PT = np.einsum("txy -> xyt",P_pos_PT)
        
f_model = open(THETA_PATH+'Theta_P={}-nfft={}.pkl'.format(P, nfft), 'wb')
g_model = open(THETA_PATH+'gamma_P={}-nfft={}-N={}.pkl'.format(P, nfft,N), 'wb')

mic_pos = np.array([[6.5, 4.49,1.5], [6.5, 4.51,1.5]])


P_pos_PTM = np.einsum("xyt,m -> xytm", P_pos_PT, np.ones(2))
r_PTM = P_pos_PTM - np.einsum("mt,yx -> yxtm",mic_pos,np.ones((ny,nx)))
Dist_PM = np.einsum("yxtm -> yxm",r_PTM**2)
Dist_flat_PM0 = np.ndarray.flatten(Dist_PM[:,:,0])
Dist_flat_PM1 = np.ndarray.flatten(Dist_PM[:,:,1])
Dist_flat_PM = np.array([Dist_flat_PM0,Dist_flat_PM1]).T
Theta_FPM = np.empty((nfft,P,M)).astype(np.complex64)

P2 = N*2
Theta_FPM2 = np.empty((nfft,P2,M)).astype(np.complex64)
gamma_NFP2 = np.zeros((N,nfft,P2))
Theta_FPM2[:,0::2,:] = Theta_FPM[:,10::36,:] * 10
Theta_FPM2[:,1::2,:] = Theta_FPM[:,28::36,:] * 10
gamma_NFP2[:,:,0::2] = gamma_NFP[:,:,10::36] 
gamma_NFP2[:,:,0::2] = gamma_NFP[:,:,28::36]

f_model = open(THETA_PATH+'Theta_P={}-nfft={}.pkl'.format(P2, nfft), 'wb')
g_model = open(THETA_PATH+'gamma_P={}-nfft={}-N={}.pkl'.format(P2, nfft,N), 'wb')


pkl.dump(Theta_FPM2, f_model)
f_model.close()
if oracle :
    pkl.dump(gamma_NFP2, g_model)
    g_model.close()
    


###Si les p choisis pour les n sources ne sont pas les bons, on peut changer order='' dans le flatten