# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:34:42 2022

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


#%%
# =============================================================================
# Audios
# =============================================================================

# concatanate audio samples to make them look long enough
wav_files = [
        ['./data/audio/in/test_piano.wav',],
        ['./data/audio/in/test_sax.wav',],
        ['./data/audio/in/test_bass.wav',]
        ]
N = len(wav_files)
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
room_dim = [8, 9, 3]

# source locations and delays
locations = [[2.5,3, 1.5], [2.5, 6,1.5] , [2.5,4.5,1.5]]
delays = [0., 0., 0.]

# create an anechoic room with sources and mics  
room = pra.ShoeBox(room_dim, fs=16000, max_order=0, absorption=0.9, sigma2_awgn=1e-8)

# add mic and good source to room
# Add silent signals to all sources
for sig, d, loc in zip(signals, delays, locations):
    room.add_source(loc, signal=sig, delay=d)

# add microphone array
room.add_microphone_array(pra.MicrophoneArray(np.c_[[6.5, 4.49,1.5], [6.5, 4.51,1.5]], room.fs))  #, [6.01 ,4.5 ], [7.0, 4.6], [3.0,3.0] , [4.3, 5.6]

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
L = 1024
hop = L // 4
win_a = pra.hamming(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)



# Observation vector in the STFT domain
X = pra.transform.stft.analysis(mics_signals.T, L, hop, win=win_a)
X = X[:,:-1,:]
t,f,m = X.shape

K=16

# =============================================================================
# X_2 = np.empty((len(wav_files),t,f,m)).astype(np.complex64)
# for n in range(len(wav_files)):
#     X_2[0] = pra.transform.stft.analysis(separate_recordings[0].T, L, hop, win=win_a)
#     X_2[1] = pra.transform.stft.analysis(separate_recordings[1].T, L, hop, win=win_a)
# X_2 = X_2.transpose(0,2,1,3)  # shape = source, f, t, m
# 
# =============================================================================

# Reference signal to calculate performance of BSS
#ref = separate_recordings[ : , 0 , :]


#%%

# =============================================================================
# FastMNMF2
# =============================================================================

def fastmnmf2(
    X,
    n_src=None,
    n_iter=30,
    n_components=16,
    mic_index=0,
    W0=None,
    accelerate=True,
    callback=None,
    Winit=None
):
    """
    Implementation of FastMNMF2 algorithm presented in

    K. Sekiguchi, Y. Bando, A. A. Nugraha, K. Yoshii, T. Kawahara, *Fast Multichannel Nonnegative
    Matrix Factorization With Directivity-Aware Jointly-Diagonalizable Spatial
    Covariance Matrices for Blind Source Separation*, IEEE/ACM TASLP, 2020.
    [`IEEE <https://ieeexplore.ieee.org/abstract/document/9177266>`_]

    The code of FastMNMF2 with GPU support and more sophisticated initialization
    is available on  https://github.com/sekiguchi92/SoundSourceSeparation

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal
    n_src: int, optional
        The number of sound sources (default None).
        If None, n_src is set to the number of microphones
    n_iter: int, optional
        The number of iterations (default 30)
    n_components: int, optional
        Number of components in the non-negative spectrum (default 8)
    mic_index: int or 'all', optional
        The index of microphone of which you want to get the source image (default 0).
        If 'all', return the source images of all microphones
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for diagonalizer Q (default None).
        If None, identity matrices are used for all frequency bins.
    accelerate: bool, optional
        If true, the basis and activation of NMF are updated simultaneously (default True)
    callback: func, optional
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    If mic_index is int, returns an (nframes, nfrequencies, nsources) array.
    If mic_index is 'all', returns an (nchannels, nframes, nfrequencies, nsources) array.
    """
    eps = 1e-10
    g_eps = 5e-2
    interval_update_Q = 2  # 2 may work as well and is faster
    interval_normalize = 10
    TYPE_FLOAT = X.real.dtype
    TYPE_COMPLEX = X.dtype

    # initialize parameter
    T = X.shape[0]
    
    X_FTM = X.transpose(1, 0, 2)
    n_freq, n_frames, n_chan = X_FTM.shape
    XX_FTMM = np.matmul(X_FTM[:, :, :, None], X_FTM[:, :, None, :].conj())
    if n_src is None:
        n_src = X_FTM.shape[2]

    if W0 is not None:
        Q_FMM = W0
    else:
        Q_FMM = np.tile(np.eye(n_chan).astype(TYPE_COMPLEX), [n_freq, 1, 1])

    g_NM = np.ones([n_src, n_chan], dtype=TYPE_FLOAT) * g_eps
    for m in range(n_chan):
        g_NM[m % n_src, m] = 1

    for m in range(n_chan):
        mu_F = (Q_FMM[:, m] * Q_FMM[:, m].conj()).sum(axis=1).real
        Q_FMM[:, m] /= np.sqrt(mu_F[:, None])

    H_NKT = np.random.rand(n_src, n_components, n_frames).astype(TYPE_FLOAT)
    if Winit is not None:
        W_NFK=Winit.astype(TYPE_FLOAT)
    else:
        W_NFK = np.random.rand(n_src, n_freq, n_components).astype(TYPE_FLOAT)            

    lambda_NFT = W_NFK @ H_NKT
    Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2
    Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM)

    def separate():
        Qx_FTM = np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)
        Qinv_FMM = np.linalg.inv(Q_FMM)
        Y_NFTM = np.einsum("nft, nm -> nftm", lambda_NFT, g_NM)

        if mic_index == "all":
            return np.einsum(
                "fij, ftj, nftj -> itfn", Qinv_FMM, Qx_FTM / Y_NFTM.sum(axis=0), Y_NFTM
            )
        elif type(mic_index) is int:
            return np.einsum(
                "fj, ftj, nftj -> tfn",
                Qinv_FMM[:, mic_index],
                Qx_FTM / Y_NFTM.sum(axis=0),
                Y_NFTM,
            )
        else:
            raise ValueError("mic_index should be int or 'all'")
            
    loss=[]

    # update parameters
    for epoch in range(n_iter):   
        print(epoch)
        
        if callback is not None and epoch % 10 == 0:
            callback(separate())
        
        #update loss
        first = np.sum(Qx_power_FTM/Y_FTM - np.log(Y_FTM))
        second = np.sum(np.log(np.abs(np.vdot(Q_FMM, Q_FMM))))
        #loss.append(-first + T/fs*second)        
        
        
        # update W and H (basis and activation of NMF)
        tmp1_NFT = np.einsum("nm, ftm -> nft", g_NM, Qx_power_FTM / (Y_FTM**2))
        tmp2_NFT = np.einsum("nm, ftm -> nft", g_NM, 1 / Y_FTM)

        numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
        denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)
        W_NFK *= np.sqrt(numerator / denominator)

        if not accelerate:
            tmp1_NFT = np.einsum("nm, ftm -> nft", g_NM, Qx_power_FTM / (Y_FTM**2))
            tmp2_NFT = np.einsum("nm, ftm -> nft", g_NM, 1 / Y_FTM)
            lambda_NFT = W_NFK @ H_NKT + eps
            Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
        denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)
        H_NKT *= np.sqrt(numerator / denominator)

        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        # update g_NM (diagonal element of spatial covariance matrices)
        numerator = np.einsum("nft, ftm -> nm", lambda_NFT, Qx_power_FTM / (Y_FTM**2))
        denominator = np.einsum("nft, ftm -> nm", lambda_NFT, 1 / Y_FTM)
        g_NM *= np.sqrt(numerator / denominator)
        Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        # udpate Q (joint diagonalizer)
        if (interval_update_Q <= 0) or (epoch % interval_update_Q == 0):
            for m in range(n_chan):
                V_FMM = (
                    np.einsum("ftij, ft -> fij", XX_FTMM, 1 / Y_FTM[..., m]) / n_frames
                )
                tmp_FM = np.linalg.solve(
                    np.matmul(Q_FMM, V_FMM), np.eye(n_chan)[None, m]
                )
                Q_FMM[:, m] = (
                    tmp_FM
                    / np.sqrt(
                        np.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM)
                    )[:, None]
                ).conj()
                Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2

        # normalize
        if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
            phi_F = np.einsum("fij, fij -> f", Q_FMM, Q_FMM.conj()).real / n_chan
            Q_FMM /= np.sqrt(phi_F)[:, None, None]
            W_NFK /= phi_F[None, :, None]

            mu_N = g_NM.sum(axis=1)
            g_NM /= mu_N[:, None]
            W_NFK *= mu_N[:, None, None]

            nu_NK = W_NFK.sum(axis=1)
            W_NFK /= nu_NK[:, None]
            H_NKT *= nu_NK[:, :, None]

            lambda_NFT = W_NFK @ H_NKT + eps
            Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2
            Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

    return separate(), loss, Q_FMM, W_NFK, H_NKT

#%%
# =============================================================================
# Separate recordings
# =============================================================================

Y,LOSS,Q,W,H = fastmnmf2(X, n_src=N , n_components=K, n_iter=100, W0=None , callback=None, Winit=None)

#%%
# =============================================================================
# Pickle the values of W and H
# =============================================================================
PATH = './data/'

f_model = open(PATH+'W_N={}-K={}-nfft={}-t={}.pkl'.format(N, K, f, t), 'wb')
pkl.dump(W, f_model)
f_model.close()

g_model = open(PATH+'H_N={}-K={}-nfft={}-t={}.pkl'.format(N, K, f, t), 'wb')
pkl.dump(H[:,:,:], g_model)
g_model.close()

#%%

# =============================================================================
# Comparaison avec les résultats de fastmnmf2 
# =============================================================================


y = pra.transform.stft.synthesis(Y[:,:,:], L, hop, win=win_s)
y = y[L - hop:, :].T

# =============================================================================
# print("Mixed signal:")
# sd.play(mixtures[-1],fs)
# status = sd.wait()
# =============================================================================

# =============================================================================
print("Separated source 0:")
sd.play(y[0], fs)
status = sd.wait()
# =============================================================================

print("Separated source 1:")
sd.play(y[1], fs)
status = sd.wait()
# =============================================================================
print("Separated source 2:")
sd.play(y[2], fs)
status = sd.wait()
