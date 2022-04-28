# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:52:47 2022

@author: MSI
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra

import soundfile as sf
import sounddevice as sd
import pickle as pkl


import time

save = True
#%%
# =============================================================================
# Audios
# =============================================================================

# concatanate audio samples to make them look long enough
wav_files = [
        ['./data/audio/in/test_bass.wav',],
        ['./data/audio/in/test_piano.wav',],
        ['./data/audio/in/test_sax.wav',]
        ]

signals = [ np.concatenate([wavfile.read(f)[1].astype(np.float32)
        for f in source_files])
for source_files in wav_files ]

#%%

# =============================================================================
# Normalisation
# =============================================================================


for i in range(len(signals)):
    signals[i] /= np.max(np.abs(signals[i]))
    
#%%
# =============================================================================
# ROOM
# =============================================================================

# Room 4m by 6m
room_dim = [8, 9]

# source locations and delays
locations = [[2.5,3], [2.5, 6], [2.5,4.5] ]
delays = [1., 0., 0.5]

# create an anechoic room with sources and mics  
room = pra.ShoeBox(room_dim, fs=16000, max_order=0, absorption=0.9, sigma2_awgn=1e-8)

# add mic and good source to room
# Add silent signals to all sources
for sig, d, loc in zip(signals, delays, locations):
    room.add_source(loc, signal=sig, delay=d)

# add microphone array
room.add_microphone_array(pra.MicrophoneArray(np.c_[[6.5, 4.49], [6.5, 4.51]], room.fs))  #, [6.01 ,4.5 ], [7.0, 4.6], [3.0,3.0] , [4.3, 5.6]


#%%

# =============================================================================
# Mix the microphone recordings to simulate the observed signals by the microphone array in the frequency domain.
# To that end, we apply the STFT transform as explained in STFT.
# =============================================================================

# Simulate
# The premix contains the signals before mixing at the microphones
# shape=(n_sources, n_mics, n_samples)
separate_recordings = room.simulate(return_premix=True)
separate_recordings /= np.max(separate_recordings)

# Mix down the recorded signals (n_mics, n_samples)
# i.e., just sum the array over the sources axis
mics_signals = np.sum(separate_recordings, axis=0)
# Normalize
mics_signals /= np.max(mics_signals)



# STFT parameters
L = 2048
hop = L // 4
win_a = pra.hamming(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

# Observation vector in the STFT domain
X = pra.transform.stft.analysis(mics_signals.T, L, hop, win=win_a)
X = X.transpose(1,0,2)
X = X[:-1,:,:]

# Reference signal to calculate performance of BSS
ref = separate_recordings[ : , 0 , :]


#%%

# =============================================================================
# Save mixture
# =============================================================================

save_path = './data/audio/out/'

nfft = L // 2

f_model = open(save_path+'mixture_nfft={}.pkl'.format(nfft), 'wb')

if save:
    pkl.dump(X, f_model)
    f_model.close()