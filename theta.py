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


nx, ny = (6, 6)   # 6*6=P//2
x = np.linspace(1.5, 3, nx)
y = np.linspace(2, 7, ny)
xv, yv = np.meshgrid(x, y)

P_pos_PT = np.array([xv,yv,1.5 * np.ones((nx,ny))])
P_pos_PT = np.einsum("txy -> xyt",P_pos_PT)
        
f_model = open(THETA_PATH+'Theta_P={}-nfft={}.pkl'.format(P, nfft), 'wb')
g_model = open(THETA_PATH+'gamma_P={}-nfft={}-N={}.pkl'.format(P, nfft,N), 'wb')

mic_pos = np.array([[6.5, 4.49,1.5], [6.5, 4.51,1.5]])


P_pos_PTM = np.einsum("xyt,m -> xytm", P_pos_PT, np.ones(2))
r_PTM = P_pos_PTM - np.einsum("mt,xy -> xytm",mic_pos,np.ones((nx,ny)))
Dist_PM = np.einsum("xytm -> xym",r_PTM**2)
Dist_flat_PM0 = np.ndarray.flatten(Dist_PM[:,:,0])
Dist_flat_PM1 = np.ndarray.flatten(Dist_PM[:,:,1])
Dist_flat_PM = np.array([Dist_flat_PM0,Dist_flat_PM1]).T
Theta_FPM = np.empty((nfft,P,M)).astype(np.complex64)

if oracle:
    gamma_NFP = np.zeros((N,nfft,P))

for f in range(nfft):
    Theta_FPM[f,:P//2,:] = 1/np.sqrt(Dist_flat_PM)*np.exp(-1j*Dist_flat_PM*f*fs/2/c)
    Theta_FPM[f,P//2:,:] = -1/np.sqrt(Dist_flat_PM)*np.exp(+1j*np.sqrt(Dist_flat_PM)*f*fs/2/c)
    if oracle:
        gamma_NFP[0,f,10] = 1.   # source 0 : (x = 2.5 , y = 3 , z = 1.5)
        gamma_NFP[0,f,46] = 1.   # gamma symetric, p_1 + 36 = p_1
        gamma_NFP[1,f,28] = 1.   # source 1 : (x = 2.5 , y = 6 , z = 1.5)
        gamma_NFP[1,f,64] = 1.


pkl.dump(Theta_FPM, f_model)
f_model.close()
if oracle :
    pkl.dump(gamma_NFP, g_model)
    g_model.close()
    


###Si les p choisis pour les n sources ne sont pas les bons, on peut changer order='' dans le flatten