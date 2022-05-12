# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:40:32 2022

@author: MSI
"""

import numpy as np
import os
import librosa
import soundfile as sf
import pickle as pkl
import ipdb

from jd_alpha_SpatialNMF import Alpha_MNMF

import argparse

import glob as glob

nfft = 512

parser = argparse.ArgumentParser()
parser.add_argument(         '--gpu', type= int, default=     0, help='GPU ID')
parser.add_argument(       '--n_fft', type= int, default=  1025, help='number of frequencies')
parser.add_argument(    '--n_speaker', type= int, default=    3, help='number of speaker')
parser.add_argument(    '--n_mic', type= int, default=    2, help='number of microphones')
parser.add_argument(     '--n_basis', type= int, default=     8, help='number of basis')
parser.add_argument( '--n_iteration', type= int, default=   2, help='number of iteration')
parser.add_argument( '--n_inter', type= int, default=  200, help='number of intervals')
parser.add_argument( '--alpha',   dest='alpha', type=float, default=1.8,  help='Gaussian case (alpha=2)')
parser.add_argument( '--seed',   dest='seed', type=int, default=0,  help='random seed for experiments')
parser.add_argument('--data', type=str, default='dev', help='available: dev or test')
parser.add_argument('--nb_file', type=int, default=1, help='nb of file to separate')
parser.add_argument('--n_Th', type=int, default=72, help='number of sphere sampling')
parser.add_argument('--id_min', type=int, default=0, help='for chunk the file')
parser.add_argument('--id_max', type=int, default=200, help='for chunk the file')
parser.add_argument('--type', type=str, default="anechoic", help='reverb or anechoic')
parser.add_argument('--update_psi',   dest='update_psi', action='store_true',  help='updating Psi matrix or not ?')

args = parser.parse_args()

for id_file in range(args.id_min, args.id_max):
    # if you have a gpu, it will use cupy instead of numpy
    if args.gpu <= 0:
        import numpy as xp
    else: 
        import cupy as xp
        print("Use GPU " + str(args.gpu))
        xp.cuda.Device(args.gpu).use()
    
    SAVE_PATH = os.path.join("./data/audio/out/",
                                 "{}_{}/alpha={}/".format(args.type,
                                                          args.n_speaker,
                                                          args.alpha))
    
    # Unpickle the mixture created by mixture_simulation.py
    fileObject2 = open('./data/audio/out/mixture_nfft={}.pkl'.format(nfft), 'rb')
    unpickler = pkl.Unpickler(fileObject2)
    mix_spec = unpickler.load().astype(np.complex64)
    fileObject2.close()
    
    if args.update_psi:
        method_name = 'Alpha-Psi_MNMF'
    else:
        method_name = 'Alpha_MNMF'
    
    
    # Solver that save parameters if True, save a wave file if True, at the path save_Path
    Separater = Alpha_MNMF(alpha=args.alpha,
                                       n_basis=args.n_basis, n_source= args.n_speaker,
                                       nb_Theta=args.n_Th, seed=args.seed,
                                       xp=xp, acoustic_model='far',
                                       update_psi=args.update_psi)
    Separater.load_spectrogram(mix_spec)
    Separater.file_id = id_file
    Separater.solve(n_iteration=args.n_iteration, save_likelihood=False,
                                save_parameter=False, save_wav=True,
                                save_path=SAVE_PATH,
                                interval_save_parameter=args.n_inter)