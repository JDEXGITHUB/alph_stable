# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 14:43:07 2022

@author: MSI
"""
#import librosa
import numpy as np
import pickle as pkl
import pyroomacoustics as pra

import soundfile as sf

#import ipdb

n_source = 2
n_mic = 2

save_fileName = "./data/audio/in/-Alpha_MNMF-oracleM={}-S={}-K=16-it=500-ID=0".format(n_mic, n_source)

n_freq = 512
hop_length = int((n_freq - 1) / 2)


fileObject2 = open(save_fileName + '-N=1.pkl', 'rb')
unpickler = pkl.Unpickler(fileObject2)
Y_NFTM = unpickler.load().astype(np.complex64)
fileObject2.close()


L = 1023
hop = L // 4 
win_a = pra.hamming(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

for n in range(n_source):
    print("n=",n)
    for m in range(n_mic):
        print("m=",m)
        tmp = pra.transform.stft.synthesis(Y_NFTM[n,:,:,m].T, L, hop, win=win_s)
        if n == 0 and m == 0:
            separated_signal = np.zeros([n_source, len(tmp), n_mic])
        separated_signal[n, :, m] = tmp
    separated_signal /= 2 #np.max(np.abs(separated_signal)) 
    print("taille signaux separes",separated_signal.shape)
    

for n in range(n_source):
    sf.write(save_fileName + "-N={}.wav".format(n), separated_signal[n,:,:] , 44100)