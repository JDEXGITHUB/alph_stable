#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import librosa
import soundfile as sf
import pickle as pkl

from alpha_SpatialNMF import Alpha_MNMF
try:
    FLAG_GPU_Available = True

except:
    print("---Warning--- You cannot use GPU acceleration because chainer or cupy is not installed")


if __name__ == "__main__":
    import argparse

    import glob as glob

    parser = argparse.ArgumentParser()
    parser.add_argument(         '--gpu', type= int, default=     0, help='GPU ID')
    parser.add_argument(       '--n_fft', type= int, default=  1024, help='number of frequencies')
    parser.add_argument(    '--n_speaker', type= int, default=    2, help='number of speaker')
    parser.add_argument(    '--n_mic', type= int, default=    2, help='number of microphones')
    parser.add_argument(     '--n_basis', type= int, default=     8, help='number of basis')
    parser.add_argument( '--n_iteration', type= int, default=   100, help='number of iteration')
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
    DIR_PATH_SOURCES = []
    PATHS = []
    for n in range(args.n_speaker):
        DIR_PATH_SOURCES.append("/media/mafontai/SSD 2/data/speech_separation/linear/{}_{}/s{}/".format(args.n_speaker, args.type, n+1))
        tmp_path = glob.glob(os.path.join(DIR_PATH_SOURCES[n], "*.wav"))
        PATHS.append(tmp_path)

    SAVE_PATH = os.path.join("/home/mafontai/Documents/project/git_project/"
                             "speech_separation/alpha_MNMF/results/",
                             "{}_{}/alpha={}/".format(args.type,
                                                      args.n_speaker,
                                                      args.alpha))
    for id_file in range(args.id_min, args.id_max):

        if args.gpu < 0:
            import numpy as xp
        else:
            import cupy as xp
            print("Use GPU " + str(args.gpu))
            xp.cuda.Device(args.gpu).use()

        sizes = []
        for n in range(args.n_speaker):
            name_file = PATHS[n][id_file]
            path_file = os.path.join(DIR_PATH_SOURCES[n], name_file)
            tmp_wav, fs = sf.read(path_file)
            sizes.append(tmp_wav.shape[0])
        length = max(sizes)
        src_wav = np.zeros((args.n_speaker, args.n_mic, length), dtype=xp.float32)

        for n in range(args.n_speaker):
            name_file = PATHS[n][id_file]
            path_file = os.path.join(DIR_PATH_SOURCES[n], name_file)
            tmp_wav, fs = sf.read(path_file)
            src_wav[n, :args.n_mic, :tmp_wav.shape[0]] = tmp_wav[:, :args.n_mic].T
            for m in range(args.n_mic):
                tmp = librosa.core.stft(np.asfortranarray(src_wav[n, m]),
                                        n_fft=args.n_fft,
                                        hop_length=int(args.n_fft/4))
                if m == 0 and n == 0:
                    src_spec = np.zeros([args.n_speaker, tmp.shape[0],
                                         tmp.shape[1], args.n_mic], dtype=xp.complex64)
                src_spec[n, :, :, m] = tmp
        mix_spec = src_spec.sum(axis=0)
        id = id_file
        if args.update_psi:
            method_name = 'Alpha-Psi_MNMF'
        else:
            method_name = 'Alpha_MNMF'
        file_path = os.path.join(SAVE_PATH,
                                 "{}-likelihood-interval=1-M={}-S={}-K={}-it={}-ID={}.pic".format(method_name, str(args.n_mic), str(args.n_speaker), str(args.n_basis), str(args.n_iteration), id))
        if os.path.exists(file_path):
            print( "{}-M={}-S={}-K={}-it={}-ID={}-N=0.wav is done !".format(method_name, str(args.n_mic), str(args.n_speaker), str(args.n_basis), str(args.n_iteration), id))
            pass
        # file_path = os.path.join(SAVE_PATH,
        #                          "{}-M={}-S={}-K={}-it={}-ID={}-N=0.wav".format(method_name, str(args.n_mic), str(args.n_speaker), str(args.n_basis), str(args.n_iteration), id))
        # if os.path.exists(file_path):
        #     print( "{}-M={}-S={}-K={}-it={}-ID={}-N=0.wav is done !".format(method_name, str(args.n_mic), str(args.n_speaker), str(args.n_basis), str(args.n_iteration), id))
        #     pass

        else:
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            Separater = Alpha_MNMF(alpha=args.alpha,
                                   n_basis=args.n_basis, n_source= args.n_speaker,
                                   nb_Theta=args.n_Th, seed=args.seed,
                                   xp=xp, acoustic_model='far',
                                   update_psi=args.update_psi)
            Separater.load_spectrogram(mix_spec)
            Separater.file_id = id_file
            Separater.solve(n_iteration=args.n_iteration, save_likelihood=True,
                            save_parameter=True, save_wav=True,
                            save_path=SAVE_PATH,
                            interval_save_parameter=args.n_inter)
