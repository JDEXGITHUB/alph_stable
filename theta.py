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
c = 340     # velocity of sound in air (in m/s)
fs = 44100  # in Hertz
N = 2       #number of sources

cas = 1

# cas = 0 le maillage et valeurs de theta et gamma qui empiriquement ont bien marché
# cas  1 ou 2 selon le maillage que l'on souhaite , pour faire des tests

if cas == 1:
    nx, ny = (4, 9)   # 6*6=P//2
    x = np.linspace(1.5, 3, nx)  #  np.linspace(1, 4.5,nx) 
    y =  np.linspace(2.5, 6.5, ny) #np.linspace(2.5,6,ny) 

if cas == 0 or cas == 2: 
    nx, ny = (6, 6)   # 6*6=P//2
    x = np.linspace(1.5, 3, nx)
    y = np.linspace(2, 7, ny)


xv, yv = np.meshgrid(x, y)

P_pos_PT = np.array([xv,yv,1.5 * np.ones((ny,nx))])  #1.5m de hauteur sont positionnés les instruments
P_pos_PT = np.einsum("txy -> xyt",P_pos_PT)
        


mic_pos = np.array([[6.5, 4.49,1.5], [6.5, 4.51,1.5]])


P_pos_PTM = np.einsum("xyt, m -> xytm", P_pos_PT, np.ones(2))
r_PTM = P_pos_PTM - np.einsum("mt, yx -> yxtm",mic_pos,np.ones((ny,nx)))
Dist_PM = np.einsum("yxtm -> yxm",r_PTM ** 2)
Dist_flat_PM0 = np.ndarray.flatten(Dist_PM[:,:,0])
Dist_flat_PM1 = np.ndarray.flatten(Dist_PM[:,:,1])
Dist_flat_PM = np.array([Dist_flat_PM0,Dist_flat_PM1]).T
Theta_FPM = np.empty((nfft,P,M)).astype(np.complex64)

gamma_NFP = np.zeros((N,nfft,P))


if cas == 0:
    for f in range(nfft):
        Theta_FPM[f,:P//2,:] = np.exp(-1j * Dist_flat_PM * f * fs / 2 / c)
        Theta_FPM[f,P//2:,:] = -np.exp(1j * np.sqrt(Dist_flat_PM) * f * fs / 2 / c)
        gamma_NFP[0,f,10] = 1.   # source 0 : (x = 2.5 , y = 3 , z = 1.5)
        gamma_NFP[0,f,46] = 1.   # gamma symetric, p_1 + 36 = p_1
        gamma_NFP[1,f,28] = 1.   # source 1 : (x = 2.5 , y = 6 , z = 1.5)
        gamma_NFP[1,f,64] = 1.

# =============================================================================
if cas == 1:
    for f in range(nfft):
        Theta_FPM[f,:P//2,:] = np.exp(-1j * np.sqrt(Dist_flat_PM) * f * fs / 2 / c / 512)
        Theta_FPM[f,P//2:,:] = -np.exp(-1j * Dist_flat_PM * f * fs / 2 / c / 512)
        gamma_NFP[0,f,6] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[6,0]) + np.sqrt(Dist_flat_PM[6,1])))   # source 0 : (x = 2.5 , y = 3 , z = 1.5)
        gamma_NFP[0,f,42] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[6,0]) + np.sqrt(Dist_flat_PM[6,1])))   # gamma symetric, p_1 + 36 = p_1
        gamma_NFP[1,f,30] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[30,0]) + np.sqrt(Dist_flat_PM[30,1])))   # source 1 : (x = 2.5 , y = 6 , z = 1.5)
        gamma_NFP[1,f,66] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[30,0]) + np.sqrt(Dist_flat_PM[30,1])))
# =============================================================================
if cas == 2:
    for f in range(nfft):
        Theta_FPM[f,:P//2,:] = np.exp(-1j * Dist_flat_PM * f * fs / 2 / c / 512)
        Theta_FPM[f,P//2:,:] = - np.exp(-1j * np.sqrt(Dist_flat_PM) * f * fs / 2 / c / 512)
        #gamma_NFP[0,f,9] = 1 #/ (0.5 * (np.sqrt(Dist_flat_PM[9,0]) + np.sqrt(Dist_flat_PM[9,1])))   # source 0 : (x = 2.5 , y = 3 , z = 1.5)
        #gamma_NFP[0,f,45] = 1 #/ (0.5 * (np.sqrt(Dist_flat_PM[9,0]) + np.sqrt(Dist_flat_PM[9,1])))   # gamma symetric, p_1 + 36 = p_1
        gamma_NFP[0,f,10] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[10,0]) + np.sqrt(Dist_flat_PM[10,1])))   # source 0 : (x = 2.5 , y = 3 , z = 1.5)
        gamma_NFP[0,f,46] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[10,0]) + np.sqrt(Dist_flat_PM[10,1])))   # gamma symetric, p_1 + 36 = p_1
#        gamma_NFP[1,f,15] = gamma_NFP[1,f,51] = gamma_NFP[1,f,52] = gamma_NFP[1,f,16] = 1
        #gamma_NFP[1,f,21] = gamma_NFP[1,f,22] = gamma_NFP[1,f,57] = gamma_NFP[1,f,58] = 1
        gamma_NFP[1,f,28] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[28,0]) + np.sqrt(Dist_flat_PM[28,1])))   # source 1 : (x = 2.5 , y = 6 , z = 1.5)
        gamma_NFP[1,f,64] = 1 / (0.5 * (np.sqrt(Dist_flat_PM[28,0]) + np.sqrt(Dist_flat_PM[28,1])))    
#        gamma_NFP[2,f,29] = 1 #/ (0.5 * (np.sqrt(Dist_flat_PM[29,0]) + np.sqrt(Dist_flat_PM[29,1])))   # source 1 : (x = 2.5 , y = 6 , z = 1.5)
#        gamma_NFP[2,f,65] = 1 #/ (0.5 * (np.sqrt(Dist_flat_PM[29,0]) + np.sqrt(Dist_flat_PM[29,1])))


gamma_NFP /= gamma_NFP.max()
f_model = open(THETA_PATH + 'Theta_P={}-nfft={}.pkl'.format(P, nfft), 'wb')
g_model = open(THETA_PATH + 'gamma_P={}-nfft={}-N={}.pkl'.format(P, nfft, N), 'wb')

pkl.dump(Theta_FPM, f_model)
f_model.close()
pkl.dump(gamma_NFP, g_model)
g_model.close()



###Si les p choisis pour les n sources ne sont pas les bons, on peut changer order='' dans le flatten


#%%


P=36

h_model = open(THETA_PATH+'1Theta_P={}-nfft={}.pkl'.format(P, nfft), 'wb')
i_model = open(THETA_PATH+'1gamma_P={}-nfft={}-N={}.pkl'.format(P, nfft,N), 'wb')

pkl.dump(Theta_FPM[:,:P,:], h_model)
h_model.close()
pkl.dump(gamma_NFP[:,:,:P], i_model)
i_model.close()