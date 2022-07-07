# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:49:49 2022

@author: MSI
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import mir_eval
from mir_eval.separation import bss_eval_sources

import pickle as pkl


path = "./data/audio/evaluators/"

n_source = 2
n_mic = 2
n_freq = 512
hop_length = int((n_freq - 1) / 2)
L = 1023
hop = L // 4 
win_a = pra.hamming(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)
SDR0, SIR0 , SAR0, PERM0 = [], [], [], []
SDR1, SIR1 , SAR1, PERM1 = [], [], [], []

fileObject5 = open("./data/audio/out/mixture_nfft=512.pkl", 'rb')
unpickler = pkl.Unpickler(fileObject5)
ref = unpickler.load()
fileObject5.close()
ref = np.asarray(ref)


# Callback function to monitor the convergence of the algorithm
def convergence_callback(Y):
    global SDR, SIR, SAR
    y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
    y = y[L - hop:, :].T
    m = np.minimum(y.shape[1], ref.shape[1])
    sdr, sir, sar, perm = bss_eval_sources(ref[:,:m], y[:,:m])
    SDR.append(sdr)
    SIR.append(sir)
    SAR.append(sar)

ref = ref.transpose(0,2,1,3)
ref_0 = pra.transform.stft.synthesis(ref.sum(axis=-1)[0,:700], L, hop, win=win_s)
ref_1 = pra.transform.stft.synthesis(ref.sum(axis=-1)[1,:700], L, hop, win=win_s)

liste = np.linspace(0,2900,30)
for i in liste:
    fileObject3 = open(path + 'Alpha_MNMF-iteration={}.pkl'.format(np.int(i)), 'rb')
    unpickler = pkl.Unpickler(fileObject3)
    lambda_nft = unpickler.load()
    fileObject3.close()
    lambda_NFT = np.asarray(lambda_nft)
    lambda_NTF = lambda_NFT.transpose(0,2,1)
    Y0 = pra.transform.stft.synthesis(lambda_NTF[0], L, hop, win=win_s)
    Y1 = pra.transform.stft.synthesis(lambda_NTF[1], L, hop, win=win_s)
    
    sdr0, sir0, sar0, perm0 = mir_eval.separation.bss_eval_sources(ref_0, Y0)
    sdr1, sir1, sar1, perm1 = mir_eval.separation.bss_eval_sources(ref_1, Y1)

    SDR0.append(sdr0)
    SIR0.append(sir0)
    SAR0.append(sar0)
    PERM0.append(perm0)    
    SDR1.append(sdr1)
    SIR1.append(sir1)
    SAR1.append(sar1)
    PERM1.append(perm1)
    
    
plt.plot(liste, SDR0, label='SDR (source0)')
#plt.plot(liste, SIR0, label='SIR (source0)')
plt.plot(liste, SAR0, label='SAR (source0)')
#plt.plot(liste, PERM0, label='PERM (source0)')
plt.plot(liste, SDR1, label='SDR (source1)')
#plt.plot(liste, SIR1, label='SIR (source1)')
plt.plot(liste, SAR1, label='SAR (source1)')
#plt.plot(liste, PERM1, label='PERM (source1)')
plt.legend()