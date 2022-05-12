# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:13:27 2022

@author: Jean-Daniel PASCAL
"""

from progressbar import progressbar

import h5py
import numpy as np
import pickle as pkl
import ipdb

import librosa
import soundfile as sf

import gc
import matplotlib.pyplot as plt

class Alpha_MNMF():
    def __init__(self, alpha=1.8, n_source=2, n_basis=8, nb_Theta=72, seed=1,
                 xp=np, acoustic_model='far', update_psi=False, oracle=False):
        """ Compute audio source separation based on Approximate LL of alpha-stable model

        Parameters:
        -----------
            alpha: int
                characteristic exponent
            nb_Theta: int
                nb of direction of arrival for
                the frequency characteristic theta (P)
            seed: int
                random seed to ensure reproducibility
            xp: 'np' or 'cp'
                cupy ('cp') is a GPU version of numpy ('np').
            acoustic_model: 'far' or 'near'
                farfield or nearfield assumption

        """
        super(Alpha_MNMF, self).__init__()
        self.eps = 1e-5
        self.alpha = alpha  # characteristic exponent
        self.n_source = n_source
        self.n_basis = n_basis
        self.P = nb_Theta
        self.update_psi = update_psi
        self.oracle = oracle

        if self.update_psi:
            self.method_name = "Alpha-Psi_MNMF"
        else:
            self.method_name = "Alpha_MNMF"
                        
        if acoustic_model == 'far':
            self.THETA_PATH = "./data/"
        self.seed = seed
        self.nE_it = 300
        self.xp = xp
        self.ac_model = acoustic_model
        self.nfft = 512
        self.rand_s = self.xp.random.RandomState(self.seed)

        file = np.load("./data/coeff.npz")
        index = np.argmin(np.abs(file['Alpha']-self.alpha))
        (a1, a2, a3, a4) = tuple(file['Eta_den_coeff'][index])
        self.a_4 = self.xp.array([a1, a2, a3, a4])
    def load_spectrogram(self, X_FTM):
        """ load complex spectrograms

        Parameters:
        -----------
            X_FTM: self.xp.array [ F x T x M ]
                complex spectrogram of observed signals
                except for oracle case: self.xp.array [ N x F x T x M ]
        """
        if self.oracle:
            self.n_source, self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        else:
            self.n_freq, self.n_time, self.n_mic = X_FTM.shape
        self.X_FTM = self.xp.asarray(X_FTM, dtype=self.xp.complex64)

# Initialization of parameters

    def init_Theta(self):
        fileObject2 = open('./data/Theta_P={}-nfft={}.pkl'.format(self.P, self.nfft), 'rb')
        unpickler = pkl.Unpickler(fileObject2)
        Theta = unpickler.load().astype(self.xp.complex64)
        fileObject2.close()
        self.Theta_FPM = self.xp.asarray(Theta)
        self.Theta_FPM /= self.xp.linalg.norm(self.Theta_FPM, axis=-1, keepdims=True)



    def init_Psi(self):
        file0 = open('./data/Theta_P={}-nfft={}.pkl'.format(self.P, self.nfft), 'rb')
        unpickler = pkl.Unpickler(file0)
        Theta = self.xp.asarray(unpickler.load().astype(self.xp.complex64))
        #Theta = pkl.load(file0).astype(self.xp.complex64)
        file0.close()
        self.Psi_FPQ = self.xp.abs(self.xp.einsum("fpm,fqm->fpq",Theta.conj(),Theta))
        #f_model = h5py.File(self.PSI_PATH + 'Psi-nfft={}-alpha={}-M={}.hdf5'.format(self.nfft, self.alpha, self.n_mic), 'r')
        #self.Psi_FQP = self.xp.asarray(f_model['Psi_FPP']).astype(self.xp.float32)
        # phi_F = self.xp.sum(self.Psi_FPP * self.Psi_FPP, axis=(1, 2)) / self.P
        # self.Psi_FPP = self.Psi_FPP / self.xp.sqrt(phi_F)[:, None, None]

    def init_SM(self):

        # self.SM_NFP = self.xp.ones((self.n_source,
        #                             self.n_freq,
        #                             self.P)).astype(self.xp.float32)
        # self.SM_NP = self.xp.ones((self.n_source,
        #                             self.P)).astype(self.xp.float32) + self.eps
        # self.SM_NP = self.xp.zeros((self.n_source,
        #                            self.P)).astype(self.xp.float32) + 1e-2
        # self.SM_NFP = self.xp.zeros((self.n_source,
        #                             self.n_freq,
        #                             self.P)).astype(self.xp.float32) + self.eps
        # quo = self.P // self.n_source
        # for n in range(self.n_source):
        #     self.SM_NFP[n, :, n * quo] = 1
        # print(self.Psi_FPQ.shape, self.SM_NFP.shape)
        fileObject3 = open('./data/gamma_P={}-nfft={}-N={}.pkl'.format(self.P, self.nfft,self.n_source), 'rb')
        unpickler = pkl.Unpickler(fileObject3)
        gamma = unpickler.load().astype(self.xp.float32)
        fileObject3.close()
        self.SM_NFP = self.xp.asarray(gamma)
        
        self.Gn_NFP = self.xp.einsum("fpq,nfp->nfq",self.Psi_FPQ, self.SM_NFP)
        #self.Gn_NFP = (self.Psi_FPP[None] * self.SM_NFP[:, :, None]).sum(axis=-1)
        # self.Gn_NFP = (self.Psi_FPP[None] * self.SM_NP[:, None, None]).sum(axis=-1) + self.eps
        # self.Gn_NFP /= self.Gn_NFP.sum(axis=-1)[:, :, None]

    def init_WH(self):
        self.W_NFK = self.xp.abs(self.rand_s.randn(self.n_source, self.n_freq, self.n_basis)).astype(self.xp.float32)
        self.H_NKT = self.xp.abs(self.rand_s.randn(self.n_source, self.n_basis, self.n_time)).astype(self.xp.float32)
        self.lambda_NFT = self.W_NFK @ self.H_NKT
        if self.oracle:
            self.lambda_NFT = 1/self.n_mic * np.einsum("nftm -> nft", self.xp.abs(self.X_FTM)**self.alpha)


    def init_auxfunc(self):
        # Fix and never updated
        self.X_FTP = self.xp.abs(self.xp.einsum("fpm , ftm -> ftp",self.Theta_FPM.conj(),
                                 self.X_FTM)).astype(self.xp.float32)
        gc.collect()
        #mempool = self.xp.get_default_memory_pool()
        #mempool.free_all_blocks()
        self.b_3 = self.xp.asarray([float(4. - 2. * (i-1) + 2*self.n_mic) for i in range(1,4)])
        self.b_3 /= self.alpha
        self.C_1 = 2 * self.a_4[-1] / self.alpha
        self.C_2 = - (2. + self.alpha) / self.alpha
        self.C_3 = self.alpha / (2. + self.alpha)

        # Auxiliary variable
        self.Y_FTP = self.xp.einsum("nfp , nft -> ftp",self.Gn_NFP,
                      self.lambda_NFT).astype(self.xp.float32)
        self.compute_Xi()

    def compute_Xi(self):
        # Auxiliary variable
        self.Xi_FTP = 1.
        Z_FTP = 1.
        # for i in range(1,4):
        #     tmp_FTP = self.xp.abs(self.a_4[i-1] *\
        #                   self.X_FTP ** (4. - 2. * (i-1)) *\
        #                   self.Y_FTP ** (-(4. - 2. * (i-1) + 2*self.n_mic) /
        #                                   self.alpha))
        #     Z_FTP += tmp_FTP
        #     self.Xi_FTP += self.b_3[i-1] * tmp_FTP
        tmp_FTP = self.xp.exp(-self.a_4[-1] * (self.X_FTP ** 2) /
                              (self.Y_FTP ** (2. / self.alpha)))
        self.Xi_FTP *= tmp_FTP
        self.Xi_FTP /= (Z_FTP * tmp_FTP + self.eps).sum(axis=-1)[..., None].astype(self.xp.float32)


# update parameters
    def update_WH(self):
        # N x F x K x T x Pp
        num_W = (self.C_1 *\
                 self.xp.einsum("nkt,ftp,ftp,nfp -> nfk",self.H_NKT,
                 self.Y_FTP ** (self.C_2),
                 self.X_FTP ** 2,
                 self.Gn_NFP))
        den_W = self.xp.einsum("ftp,ftp,nkt,nfp -> nfk",self.Xi_FTP,
                 self.Y_FTP ** (- 1),
                 self.H_NKT,
                 self.Gn_NFP) + self.eps
        self.W_NFK *= (num_W/den_W) ** self.C_3
        self.reset_variable(type='NMF')
        # N x F x K x T x Pp

        num_H = (self.C_1 *\
                 self.xp.einsum("nfk,ftp,ftp,nfp -> nkt",self.W_NFK,
                 self.Y_FTP ** (self.C_2),
                 self.X_FTP ** 2,
                 self.Gn_NFP))
        den_H = self.xp.einsum("ftp,ftp,nfk,nfp -> nkt",self.Xi_FTP,
                 self.Y_FTP ** (- 1),
                 self.W_NFK,
                 self.Gn_NFP) + self.eps
        self.H_NKT *= (num_H/den_H) ** self.C_3
        self.reset_variable(type='NMF')

    def update_SM(self):
        # N x F x T x Pp x P
        num_SM = (self.C_1 *\
                 self.xp.einsum("nft,ftq,ftq,fqp -> nfp",self.lambda_NFT,
                 self.Y_FTP ** (self.C_2) ,
                 (self.X_FTP ** 2) ,
                 self.Psi_FPQ ** self.alpha))
        den_SM = self.xp.einsum("ftq,ftq,nft,fqp -> nfp",self.Xi_FTP,
                 self.Y_FTP ** (- 1) ,
                 self.lambda_NFT ,
                 self.Psi_FPQ ** self.alpha) + self.eps
        self.SM_NFP *= (num_SM/den_SM) ** (self.C_3)
        # self.SM_NFP /= self.xp.max(self.SM_NFP, axis=-1)[:, :, None]
        self.reset_variable(type='SM')

    def update_Psi(self):  #il manque un psi dans le den??
        # N x F x T x Pp x P
        num_Psi = (self.C_1 *\
                 self.xp.einsum("nft,ftq,ftq,nfp -> fqp",self.lambda_NFT,
                 self.Y_FTP ** (self.C_2),
                 (self.X_FTP ** 2) ,
                 self.SM_NFP))
        den_Psi = self.xp.einsum("ftq,ftq,nft,nfp,fqp -> fqp",(self.Xi_FTP,
                 self.Y_FTP ** (- 1) ,
                 self.lambda_NFT,
                 self.SM_NFP , self.Psi_FPQ ** self.alpha)) + self.eps
        self.Psi_FPQ *= (num_Psi/den_Psi) ** (self.C_3)
        self.reset_variable(type='Psi')

    def reset_variable(self, type):
        if type == 'NMF':
            self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps
            self.lambda_NFT = self.xp.clip(self.lambda_NFT, a_min=self.xp.exp(-16),
                                           a_max=self.xp.exp(16))

        elif type == 'SM' or type == 'Psi':
            self.Gn_NFP = self.xp.einsum("fpq,nfq->nfp",self.Psi_FPQ, self.SM_NFP)
            self.Gn_NFP = self.xp.einsum("fqp,nfp->nfp",self.Psi_FPQ,
                           self.SM_NFP)
            # self.Gn_NFP = (self.Psi_FPP[None] *
            #                self.SM_NP[:, None, None]).sum(axis=-1)
        elif type == 'all':
            self.lambda_NFT = self.W_NFK @ self.H_NKT + self.eps
            self.lambda_NFT = self.xp.clip(self.lambda_NFT, a_min=self.xp.exp(-16),
                                           a_max=self.xp.exp(16))
            self.Gn_NFP = self.xp.einsum("fpq,nfq->nfp", self.Psi_FPQ,
                           self.SM_NFP)
            # self.Gn_NFP = (self.Psi_FPP[None] *
            #                self.SM_NP[:, None, None]).sum(axis=-1)
        self.compute_Xi()
        self.Xi_FTP = self.xp.clip(self.Xi_FTP, a_min=self.xp.exp(-16),
                                   a_max=self.xp.exp(16))
        self.Y_FTP = self.xp.einsum("nfp,nft -> ftp",self.Gn_NFP, self.lambda_NFT)



    def normalize(self):
        # if self.update_psi:
        # phi_FP = self.xp.sum(self.Psi_FPP * self.Psi_FPP, axis=1) / self.P
        # self.Psi_FPP /= self.xp.sqrt(phi_FP)[:, None, :]
        # self.SM_NFP /= phi_FP[None]

        mu_NF = self.SM_NFP.sum(axis=-1)
        self.SM_NFP /= mu_NF[:, :, None]
        self.W_NFK *= mu_NF[..., None]

        # mu_N = self.SM_NP.sum(axis=-1)
        # self.SM_NP /= mu_N[:, None]
        # self.W_NFK *= mu_N[:, None, None]

        self.W_NFK = self.xp.clip(self.W_NFK, a_min=self.xp.exp(-16),
                                  a_max=self.xp.exp(16))
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        # self.W_NFK *= self.n_freq
        self.H_NKT *= nu_NK[:, :, None]
        self.H_NKT = self.xp.clip(self.H_NKT, a_min=self.xp.exp(-16),
                                  a_max=self.xp.exp(16))
        self.reset_variable(type='all')
#   E-Step part ########################################

    def update_P(self):
        #  N F T "M" P

        Cste = float(4 * self.xp.pi / self.P)  # integration constant, stereo case

        WTh_NFTMP = self.xp.einsum("nftml, fpl -> nftmp",self.W_NFTMM.conj(),
                     self.Theta_FPM)

        # N F T "M" M M P
        tmpI1_1 = self.xp.einsum("flkp, nftmp -> nftmlkp",self.ThTh_FMMP,
              self.xp.abs(self.Theta_FPM.transpose(0, 2, 1)[None, :, None, :]
               - WTh_NFTMP + self.eps) ** (self.alpha - 2.), optimize=True)
        tmpI1_2 = self.xp.einsum("flkp, nftmp -> nftmlkp",self.ThTh_FMMP,
              self.xp.abs(WTh_NFTMP + self.eps) ** (self.alpha - 2.))
        I1 = tmpI1_1 - tmpI1_2
        I1 = self.xp.einsum("nftmlkp, nft, nfp -> nftmlkp", I1, self.lambda_NFT, self.SM_NFP)

        cov_FTP = self.xp.einsum("nft, nfp -> ftp",self.lambda_NFT, self.SM_NFP)
        
        # N F T "M" M M P
        I2 = self.xp.einsum("nftmlkp, ftp -> nftmlkp", tmpI1_2, cov_FTP)
        self.P_NFTMMM = Cste * (I1 + I2).sum(axis=-1)

    def update_Lagrange(self):
        InvP_NFTMMM = self.xp.linalg.inv(self.P_NFTMMM)
        Inv_FTMMM = self.xp.linalg.inv(InvP_NFTMMM.sum(axis=0))
        if self.xp == "cp":
            InvP_NFTMMM = self.xp.array(InvP_NFTMMM)
            Inv_FTMMM = self.xp.array(Inv_FTMMM)
        Id_FTMM =  self.xp.einsum("nftmlk,ftmlk->ftml",InvP_NFTMMM,Inv_FTMMM)
        self.La_FTMM += self.xp.einsum("ftmlk, ftmk -> ftml",Inv_FTMMM ,
                         (self.W_NFTMM.sum(axis=0) - Id_FTMM))

    def update_W(self):
        Cste = float(4 * self.xp.pi / self.P)  # integration constant
        #WTh_NFTMP = (self.W_NFTMM[..., None].conj() * self.Theta_FPM.transpose(0, 2, 1)[None, :, None, None]).sum(axis=-2)
        
        WTh_NFTMP = self.xp.einsum("nftml,fpl -> nftmp",self.W_NFTMM.conj(),self.Theta_FPM)

        # N F T M "M" P -> N F T "M" M
        R_NFTMM = Cste * self.xp.einsum("fpl,fpm,nftmp,nft,nfp->nftml",self.Theta_FPM, self.Theta_FPM.conj(), 
                (self.Theta_FPM.transpose(0, 2, 1)[None, :, None, :]
                - WTh_NFTMP + self.eps) ** (self.alpha - 2.)  ,
                self.lambda_NFT , self.SM_NFP)

        InvP_NFTMMM = self.xp.linalg.inv(self.P_NFTMMM)
        self.W_NFTMM = self.xp.einsum("nftmlk,nftmk -> nftml",InvP_NFTMMM , (R_NFTMM - self.La_FTMM[None,:]))

    def E_Step(self):
        del self.Psi_FPQ, self.Xi_FTP
        gc.collect()
        mempool = self.xp.get_default_memory_pool()
        mempool.free_all_blocks()

        # Init variables
        self.W_NFTMM = self.xp.ones((self.n_source, self.n_freq, self.n_time, self.n_mic, self.n_mic)).astype(self.xp.complex64)
        self.W_NFTMM *= (self.xp.eye(self.n_mic)/self.n_source)[None, None, None]
        self.La_FTMM = self.xp.zeros((self.n_freq, self.n_time, self.n_mic, self.n_mic)).astype(self.xp.complex64)
        self.P_NFTMMM = self.rand_s.rand(self.n_source, self.n_freq, self.n_time, self.n_mic, self.n_mic, self.n_mic).astype(self.xp.complex64)
        self.ThTh_FMMP = self.xp.einsum("fpm,fpl->fmlp",self.Theta_FPM , self.Theta_FPM.conj())
        Id_NFTMMM = self.xp.ones((self.n_source, self.n_freq, self.n_time,
                                  self.n_mic, self.n_mic, self.n_mic)) *\
                    self.xp.eye(self.n_mic)[None, None, None, None]
        for it in range(self.nE_it):
            print("filtrage iteration = {}".format(it))
            self.update_P()
            self.P_NFTMMM += 1e-3 * Id_NFTMMM
            self.P_NFTMMM[self.xp.isnan(self.P_NFTMMM)] = self.eps
            #ipdb.set_trace()
            self.update_Lagrange()
            self.La_FTMM[self.xp.isnan(self.La_FTMM)] = self.eps
            self.update_W()
            self.W_NFTMM[self.xp.isnan(self.W_NFTMM)] = self.eps
        #ipdb.set_trace()
        self.Y_NFTM = self.xp.einsum("nftml,ftl->nftm", self.W_NFTMM.conj() , self.X_FTM)


    def calculate_log_likelihood(self):
        tmp_FTP = (self.a_4[0] * (self.X_FTP ** 4) / (self.Y_FTP ** ((4 + 2 * self.n_mic) / self.alpha)) +\
                   self.a_4[1] * (self.X_FTP ** 2) / (self.Y_FTP ** ((2 + 2 * self.n_mic) / self.alpha)) +\
                   self.a_4[2] * 1. / (self.Y_FTP ** ((2 * self.n_mic) / self.alpha))) *\
            self.xp.exp(-self.a_4[3] * (self.X_FTP ** 2) / (self.Y_FTP ** (2. / self.alpha)))

        ll_value = (self.xp.log(tmp_FTP.sum(axis=-1) + self.eps)).sum()

        return self.convert_to_NumpyArray(ll_value)

    def make_filename_suffix(self):
        self.filename_suffix = "M={}-S={}-K={}-it={}".format(self.n_mic, self.n_source, self.n_basis, self.n_iteration)
        if self.oracle == True:
            self.filename_suffix = "oracle" + self.filename_suffix

        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        return self.filename_suffix

    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def solve(self, n_iteration=100, save_likelihood=False,
              save_parameter=False, save_wav=True,
              save_path="./", interval_save_parameter=10):
        """
        Parameters:
            save_likelihood: boolean
                save likelihood and lower bound or not
            save_parameter: boolean
                save parameter or not
            save_wav: boolean
                save intermediate separated signal or not
            save_path: str
                directory for saving data
            interval_save_parameter: int
                interval of saving parameter
        """

        # Initialization
        self.n_iteration = n_iteration
        self.save_path = save_path
        self.init_Theta()
        self.init_Psi()
        self.init_SM()
        self.init_WH()
        if self.oracle:
            self.X_FTM = self.xp.einsum('nftp -> ftp',self.X_FTM)
        self.init_auxfunc()
        self.make_filename_suffix()

        ll_array = []
        f_, ax = plt.subplots(2, 2)
        end = -1
        for it in range(self.n_iteration):
            if self.oracle:
                break
            print(it)
            #ipdb.set_trace()
            self.ac = it
            self.update_WH()
            if self.oracle == False:
                self.update_SM()
            if self.update_psi:
                self.update_Psi()
            self.normalize()
            ax[0, 0].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.lambda_NFT[0])**2)), origin="lower", label='lambda')
            ax[0, 1].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.lambda_NFT[1])**2)), origin="lower", label='lambda')
            ax[1, 0].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.SM_NFP[0])**2)), origin="lower",label='gamma')
            ax[1, 1].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.SM_NFP[1])**2)), origin="lower",label='gamma')
            # ax[1, 0].plot(self.convert_to_NumpyArray(self.SM_NP[0]))
            # ax[1, 1].plot(self.convert_to_NumpyArray(self.SM_NP[1]))
            if save_likelihood and ((it+1) % interval_save_parameter == 0) and ((it+1) != self.n_iteration):
                ll_res = self.calculate_log_likelihood()
                ll_array.append(ll_res)
                # plt.plot(ll_array)
                # plt.show()
                plt.savefig(save_path + "test{}.png".format(it))
            end += 1 

        if save_likelihood and (end+1 == self.n_iteration):
            ll_res = self.calculate_log_likelihood()
            ll_array.append(ll_res)
            plt.plot(ll_array)
            plt.show()
            # plt.savefig(save_path + "{}-likelihood-interval={}-{}.png".format(self.method_name, interval_save_parameter, self.filename_suffix))
            # pkl.dump(ll_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
        if save_wav and ((end+1) == self.n_iteration):
            self.E_Step()
            self.save_separated_signal(save_path+"{}-{}".format(self.method_name, self.filename_suffix))

        if save_parameter:
            with h5py.File(save_path + "spatial-measure-{}-{}.hdf5".format(self.method_name, self.filename_suffix), 'w') as file_obj:
                file_obj.create_dataset(
                    'SM_NFP',
                    data=self.convert_to_NumpyArray(self.SM_NFP).astype(np.float32),
                    dtype=np.float32,
                    compression='lzf')

    def save_separated_signal(self, save_fileName="sample.wav"):
        self.Y_NFTM = self.convert_to_NumpyArray(self.Y_NFTM)
        hop_length = int((self.n_freq - 1) / 2)
        for n in range(self.n_source):
            for m in range(self.n_mic):
                tmp = librosa.core.istft(self.Y_NFTM[n, :, :, m],
                                         hop_length=hop_length)
                if n == 0 and m == 0:
                    separated_signal = np.zeros([self.n_source, len(tmp), self.n_mic])
                    print("taille signaux separes",separated_signal.shape)
                separated_signal[n, :, m] = tmp
        separated_signal /= np.abs(separated_signal).max() * 1.2

        for n in range(self.n_source):
            sf.write(save_fileName + "-N={}.wav".format(n), separated_signal[n,:,:], 41000)
