"""
Created on Thu Apr 14 15:13:27 2022

@author: Jean-Daniel PASCAL
"""

#from progressbar import progressbar

#import h5py
import numpy as np
import pickle as pkl
import ipdb

#import librosa
import soundfile as sf
#import scipy as sc

import gc
import matplotlib.pyplot as plt
#import mir_eval.separation

from tempfile import mkdtemp
import os.path as path

#import tracemalloc

class Alpha_MNMF():
    def __init__(self, alpha=1.8, n_source=2, n_basis=8, nb_Theta=36, seed=1,
                 xp=np, acoustic_model='far', update_psi=False, oracle=False, init_parameter=True):
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
        self.init_parameter = init_parameter

        if self.update_psi:
            self.method_name = "Alpha-Psi_MNMF"
        else:
            self.method_name = "-Alpha_MNMF"
                        
        if acoustic_model == 'far':
            self.THETA_PATH = "./data/"
        self.seed = seed
        self.nE_it = 500
        self.xp = xp
        self.ac_model = acoustic_model
        self.nfft = 512
        self.rand_s = self.xp.random.RandomState(self.seed)

        file = np.load("/tsi/clusterhome/jdpascal/alph_stable/data/coeff.npz")
        index = np.argmin(np.abs(file['Alpha']-self.alpha))
        (a1, a2, a3, a4) = tuple(file['Eta_den_coeff'][index])
        self.a_4 = self.xp.array([a1, a2, a3, a4])
        
    def load_spectrogram(self, X_NFTM):
        """ load complex spectrograms

        Parameters:
        -----------
            X_NFTM: self.xp.array [ N x F x T x M ]
                complex spectrogram of observed signals
        """
        self.n_source, self.n_freq, self.n_time, self.n_mic = X_NFTM.shape
        self.X_NFTM = self.xp.asarray(X_NFTM, dtype=self.xp.complex64)

# Initialization of parameters

    def init_Theta(self):
        fileObject2 = open('/tsi/clusterhome/jdpascal/alph_stable/data/Theta_P={}-nfft={}.pkl'.format(self.P, self.nfft), 'rb')
        unpickler = pkl.Unpickler(fileObject2)
        Theta = unpickler.load().astype(self.xp.complex64)
        fileObject2.close()
        self.Theta_FPM = self.xp.asarray(Theta)
        # self.Theta_FPM /= self.xp.linalg.norm(self.Theta_FPM, axis=-1, keepdims=True)
        

    def init_Psi(self):
        file0 = open('/tsi/clusterhome/jdpascal/alph_stable/data/Theta_P={}-nfft={}.pkl'.format(self.P, self.nfft), 'rb')
        unpickler = pkl.Unpickler(file0)
        Theta = self.xp.asarray(unpickler.load().astype(self.xp.complex64))
        #Theta = pkl.load(file0).astype(self.xp.complex64)
        file0.close()
        self.Psi_FPQ = (self.xp.abs(self.xp.einsum("fpm,fqm->fpq",Theta.conj(),Theta)) ** self.alpha ).astype(self.xp.float32)


    def init_SM(self):
        fileObject3 = open('/tsi/clusterhome/jdpascal/alph_stable/data/gamma_P={}-nfft={}-N={}.pkl'.format(self.P, self.nfft,self.n_source), 'rb')
        unpickler = pkl.Unpickler(fileObject3)
        gamma = unpickler.load().astype(self.xp.float32)
        fileObject3.close()
        self.SM_NFP = self.xp.asarray(gamma) 
        # SM_NFP2 = self.xp.asarray(gamma)
        # self.SM_NFP[0] = SM_NFP2[1]
        # self.SM_NFP[1] = SM_NFP2[0]
        # del SM_NFP2
        # for n in range(self.n_source):
        #     self.SM_NFP[n] /= self.xp.max(self.SM_NFP[n]) 
        self.SM_NFP += self.eps
        self.Gn_NFP = self.xp.einsum("fpq, nfp -> nfq", self.Psi_FPQ, self.SM_NFP)

    def init_WH(self):
        self.W_NFK = self.xp.abs(self.rand_s.randn(self.n_source, self.n_freq, self.n_basis)).astype(self.xp.float32)
        self.H_NKT = self.xp.abs(self.rand_s.randn(self.n_source, self.n_basis, self.n_time)).astype(self.xp.float32)
        if self.init_parameter:
            fileObjectW = open('/tsi/clusterhome/jdpascal/alph_stable/data/W_N={}-K={}-nfft={}-t={}.pkl'.format(self.n_source, self.n_basis, self.n_freq,self.n_time), 'rb')
            unpicklerW = pkl.Unpickler(fileObjectW)
            W = unpicklerW.load().astype(self.xp.complex64)
            fileObjectW.close()
            self.W_NFK = self.xp.abs(self.xp.asarray(W)).astype(self.xp.float32)
            self.W_NFK /= self.xp.max(self.W_NFK)

            fileObjectH = open('/tsi/clusterhome/jdpascal/alph_stable/data/H_N={}-K={}-nfft={}-t={}.pkl'.format(self.n_source, self.n_basis, self.n_freq,self.n_time), 'rb')
            unpicklerH = pkl.Unpickler(fileObjectH)
            H = unpicklerH.load().astype(self.xp.complex64)
            fileObjectH.close()
            self.H_NKT = self.xp.abs(self.xp.asarray(H)).astype(self.xp.float32)
            self.H_NKT /= self.xp.max(self.H_NKT)
            del W, H, fileObjectH, fileObjectW, unpicklerH, unpicklerW
        #ipdb.set_trace()

        if self.oracle:
            self.lambda_NFT = 1 / self.n_mic * self.xp.einsum("nftm -> nft", self.xp.abs(self.X_NFTM) ** self.alpha).astype(self.xp.float32)
            # self.lambda_NFT /= self.xp.max(self.lambda_NFT)
            # ipdb.set_trace()
            # for n in range(self.n_source):
            #     self.lambda_NFT[n] /= self.xp.max(self.lambda_NFT[n]) 
        else:
            self.lambda_NFT = self.W_NFK @ self.H_NKT
        
        


    def init_auxfunc(self):
        # Fix and never updated
        self.X_FTP = self.xp.abs(self.xp.einsum("fpm , ftm -> ftp", self.Theta_FPM.conj(),
                                 self.X_FTM)).astype(self.xp.float32)
        #gc.collect()
        #mempool = self.xp.get_default_memory_pool()
        #mempool.free_all_blocks()
        self.b_3 = self.xp.asarray([float(4. - 2. * (i-1) + 2*self.n_mic) for i in range(1,4)])
        self.b_3 /= self.alpha
        self.C_1 = 2 * self.a_4[-1] / self.alpha
        self.C_2 = - (2. + self.alpha) / self.alpha
        self.C_3 = self.alpha / (2. + self.alpha)

        # Auxiliary variable
        self.Y_FTP = self.xp.einsum("nfp, fpq -> nfq",self.SM_NFP, self.Psi_FPQ).astype(self.xp.float32)
        self.Y_FTP = self.xp.einsum("nfq, nft -> ftq", self.Y_FTP, self.lambda_NFT)
        self.compute_Xi()

    def compute_Xi(self):
        # Auxiliary variable
        self.Xi_FTP = 0.
        Z_FTP = 0.
        for i in range(1,4):
            tmp_FTP = self.a_4[i-1] * self.X_FTP ** (4. - 2. * (i-1)) * (self.Y_FTP + self.eps) ** (-(4. - 2. * (i-1) + 2*self.n_mic) / self.alpha)
            Z_FTP += tmp_FTP
            self.Xi_FTP += self.b_3[i-1] * tmp_FTP
        tmp_FTP = self.xp.exp(-self.a_4[-1] * (self.X_FTP ** 2) / ((self.Y_FTP + self.eps) ** (2. / self.alpha)))
        #Z_FTP /= self.xp.max(Z_FTP)
        self.Xi_FTP *= tmp_FTP
        self.Xi_FTP /= (Z_FTP * tmp_FTP + self.eps).sum(axis=-1)[..., None].astype(self.xp.float32)
        #self.Xi_FTP /= self.xp.max(self.Xi_FTP)
        #ipdb.set_trace()
        del Z_FTP, tmp_FTP

# Update parameters
    def update_WH(self):
        # N x F x K 
        #filename = path.join(mkdtemp(), 'num_W.dat')
        # num_W = np.memmap(filename, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.n_time))
        
        num_W = (self.C_1 *\
                 self.xp.einsum("ftq, fpq, nfp -> nft",
                 (self.Y_FTP + self.eps) ** (self.C_2) * self.X_FTP ** 2,
                 self.Psi_FPQ,
                 self.SM_NFP))
        
        #filename2 = path.join(mkdtemp(), 'num_W2.dat')     
        # num_W2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.n_basis))
        num_W2 = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, num_W)
        
        #filename3 = path.join(mkdtemp(), 'den_W.dat')     
        # den_W = np.memmap(filename3, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.n_time))
        den_W = self.xp.einsum("ftq, fpq, nfp -> nft",
                self.Xi_FTP * (self.Y_FTP + self.eps) ** (- 1),
                self.Psi_FPQ,
                self.SM_NFP) + self.eps
        #filename4 = path.join(mkdtemp(), 'den_W2.dat')     
        # den_W2 = np.memmap(filename4, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.n_basis))
        den_W2 = self.xp.einsum("nkt, nft -> nfk", self.H_NKT, den_W)
        self.W_NFK *= (num_W2/den_W2) ** self.C_3
        self.reset_variable(type='NMF')

        # N x K x T 
        #filename5 = path.join(mkdtemp(), 'num_H.dat')     
        # num_H = np.memmap(filename5, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.n_time))
        num_H = (self.C_1 *\
                 self.xp.einsum("ftq, fpq, nfp -> nft",
                 (self.Y_FTP + self.eps) ** (self.C_2) * self.X_FTP ** 2,
                 self.Psi_FPQ,
                 self.SM_NFP))
        #filename6 = path.join(mkdtemp(), 'num_H2.dat')     
        # num_H2 = np.memmap(filename6, dtype='float32', mode='w+', shape=(self.n_source,self.n_basis,self.n_time))
        num_H2 =  self.xp.einsum("nfk, nft -> nkt", self.W_NFK, num_H)
        
        #filename7 = path.join(mkdtemp(), 'den_H.dat')     
        # den_H = np.memmap(filename7, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.n_time))
        den_H =  self.xp.einsum("ftq, fpq, nfp -> nft",
                 self.Xi_FTP * (self.Y_FTP + self.eps) ** (- 1),
                 self.Psi_FPQ,
                 self.SM_NFP) + self.eps
        #filename8 = path.join(mkdtemp(), 'den_H2.dat')     
        # den_H2 = np.memmap(filename8, dtype='float32', mode='w+', shape=(self.n_source,self.n_basis,self.n_time))
        den_H2 = self.xp.einsum("nfk, nft -> nkt", self.W_NFK, den_H) 
                 
        self.H_NKT *= (num_H2/den_H2) ** self.C_3
        self.reset_variable(type='NMF')
        del num_W, num_H, den_W, den_H, num_W2, num_H2, den_W2, den_H2

    def update_SM(self):
        # N x F x T x Pp x P
        #filename = path.join(mkdtemp(), 'num_SM.dat')     
        # ipdb.set_trace()
        # num_SM = np.memmap(filename, dtype='float32', mode='w+', shape=(self.n_freq,self.n_time,self.P))
        num_SM = (self.C_1 *\
                 self.xp.einsum("ftq, fqp -> ftp",
                 (self.Y_FTP + self.eps) ** (self.C_2) * self.X_FTP ** 2 ,
                 self.Psi_FPQ))
        # ipdb.set_trace()
        #filename2 = path.join(mkdtemp(), 'num_SM2.dat')     
        # num_SM2 = np.memmap(filename2, dtype='float32', mode='w+', shape=(self.n_source,self.n_freq,self.P))
        num_SM2 = self.xp.einsum("nft,ftp -> nfp", self.lambda_NFT, num_SM)
        
        #filename3 = path.join(mkdtemp(), 'den_SM.dat')     
        # den_SM = np.memmap(filename3, dtype='float32', mode='w+', shape=(self.n_freq,self.n_time,self.P))
        den_SM = self.xp.einsum("ftq, fqp -> ftp",
                 self.Xi_FTP * (self.Y_FTP + self.eps) ** (- 1) ,
                 self.Psi_FPQ) + self.eps
        # ipdb.set_trace()
        #filename4 = path.join(mkdtemp(), 'den_SM2.dat')     
        # den_SM2 = np.memmap(filename4, dtype='float32', mode='w+', shape=(self.n_freq,self.n_time,self.P))
        den_SM2 = self.xp.einsum("ftp, nft -> nfp", den_SM, self.lambda_NFT)
        
        self.SM_NFP = self.SM_NFP * (num_SM2 / den_SM2) ** (self.C_3)
        # self.SM_NFP /= self.xp.max(self.SM_NFP, axis=-1)[:, :, None]
        self.reset_variable(type='SM')
        # ipdb.set_trace()
        del num_SM, den_SM, num_SM2, den_SM2, #filename, #filename2, #filename3, #filename4

    def update_Psi(self):  #il manque un psi dans le den??
        # N x F x T x Pp x P
        #filename = path.join(mkdtemp(), 'num_Psi.dat') 
        # num_Psi = np.memmap(filename, dtype='complex64', mode='w+', shape=(self.n_freq,self.P,self.P))
        num_Psi = (self.C_1 *\
                 self.xp.einsum("nft, ftq, nfp -> fpq",
                 self.lambda_NFT,
                 (self.Y_FTP + self.eps) ** (self.C_2) * (self.X_FTP ** 2) ,
                 self.SM_NFP))
        #filename2 = path.join(mkdtemp(), 'den_Psi.dat') 
        # den_Psi = np.memmap(filename2, dtype='complex64', mode='w+', shape=(self.n_freq,self.P,self.P))
        den_Psi = self.xp.einsum("ftq, nft, nfp, fpq -> fpq",
                 self.Xi_FTP * (self.Y_FTP + self.eps) ** (- 1) ,
                 self.lambda_NFT,
                 self.SM_NFP , self.Psi_FPQ) + self.eps
        self.Psi_FPQ *= (num_Psi / den_Psi) ** (self.C_3)
        self.reset_variable(type='Psi')
        del num_Psi, den_Psi

    def reset_variable(self, type):
        if type == 'NMF':
            self.lambda_NFT = self.W_NFK @ self.H_NKT 
            #self.lambda_NFT = self.xp.clip(self.lambda_NFT, a_min=self.xp.exp(-16), a_max=self.xp.exp(16))
        elif type == 'SM' or type == 'Psi':
            self.Gn_NFP = self.xp.einsum("fpq, nfp -> nfq",self.Psi_FPQ, self.SM_NFP)
        elif type == 'all':
            self.lambda_NFT = self.xp.einsum("nfk,nkt->nft",self.W_NFK ,self.H_NKT )
            # self.lambda_NFT = self.xp.clip(self.lambda_NFT, a_min=self.xp.exp(-16), a_max=self.xp.exp(16))
            self.Gn_NFP = self.xp.einsum("fpq, nfp -> nfp", self.Psi_FPQ, self.SM_NFP)
        self.compute_Xi()
        # self.Xi_FTP = self.xp.clip(self.Xi_FTP, a_min=self.xp.exp(-16), a_max=self.xp.exp(16))
        self.Y_FTP = self.xp.einsum("nfq, nft -> ftq",self.Gn_NFP, self.lambda_NFT)

    def normalize(self):
        # if self.update_psi:
        # phi_FP = self.xp.sum(self.Psi_FPP * self.Psi_FPP, axis=1) / self.P
        # self.Psi_FPP /= self.xp.sqrt(phi_FP)[:, None, :]
        # self.SM_NFP /= phi_FP[None]

        if self.oracle == False:
            mu_NF = self.SM_NFP.sum(axis=-1)
            self.SM_NFP /= mu_NF[:, :, None]
            self.W_NFK *= mu_NF[..., None]

        # mu_N = self.SM_NP.sum(axis=-1)
        # self.SM_NP /= mu_N[:, None]
        # self.W_NFK *= mu_N[:, None, None]

        #self.W_NFK = self.xp.clip(self.W_NFK, a_min=self.xp.exp(-16), a_max=self.xp.exp(16))
        nu_NK = self.W_NFK.sum(axis=1)
        self.W_NFK /= nu_NK[:, None]
        # self.W_NFK *= self.n_freq
        self.H_NKT *= nu_NK[:, :, None]
        #self.H_NKT = self.xp.clip(self.H_NKT, a_min=self.xp.exp(-16), a_max=self.xp.exp(16))
        self.reset_variable(type='all')
#   E-Step part ########################################

    def update_P(self):
        #  N F T "M" P

        Cste = float(4 * self.xp.pi / self.P)  # integration constant, stereo case

        WTh_NFTMP = self.xp.einsum("nftml, fpl -> nftmp",self.W_NFTMM.conj(),
                     self.Theta_FPM)

        # N F T "M" M M P
        #filename = path.join(mkdtemp(), 'tmpI1.dat')     
        # tmpI1 = np.memmap(filename, dtype='complex64', mode='w+', shape=(self.n_source,self.n_freq,self.n_time,self.n_mic,self.n_mic,self.n_mic,self.P))
        tmpI1 = self.xp.einsum("flkp, nftmp -> nftmlkp",self.ThTh_FMMP,
              self.xp.abs(self.Theta_FPM.transpose(0, 2, 1)[None, :, None, :] - WTh_NFTMP + self.eps) ** (self.alpha - 2.))
        #del tmpI1
        
        #filename2 = path.join(mkdtemp(), 'tmpI2.dat')
        # tmpI2 = np.memmap(filename2, dtype='complex64', mode='w+', shape=(self.n_source,self.n_freq,self.n_time,self.n_mic,self.n_mic,self.n_mic,self.P))
        tmpI2 = self.xp.einsum("flkp, nftmp -> nftmlkp",self.ThTh_FMMP,
              self.xp.abs(WTh_NFTMP + self.eps) ** (self.alpha - 2.))
        cov_FTP = self.xp.einsum("nft, nfp -> ftp", self.lambda_NFT, self.SM_NFP)
        
        # N F T "M" M M P
        tmpI2 = self.xp.einsum("nftmlkp, ftp -> nftmlkp", tmpI2, cov_FTP)
        #del cov_FTP

        #tmpI1 = np.memmap(filename, dtype='complex64', mode='r+', shape=(self.n_source,self.n_freq,self.n_time,self.n_mic,self.n_mic,self.n_mic,self.P))
        #tmpI1 = self.xp.array(tmpI1)
        tmpI1 = tmpI1 - tmpI2

        #del tmpI2
        tmpI1 = self.xp.einsum("nftmlkp, nft, nfp -> nftmlkp", tmpI1, self.lambda_NFT, self.SM_NFP)
        # N F T "M" M M P
        #tmpI2 = np.memmap(filename2, dtype='complex64', mode='r+', shape=(self.n_source,self.n_freq,self.n_time,self.n_mic,self.n_mic,self.n_mic,self.P))
        #tmpI2 = self.xp.array(tmpI2)
        self.P_NFTMMM = Cste * (tmpI1 + tmpI2).sum(axis=-1)
        
        del Cste, tmpI1, tmpI2, WTh_NFTMP

    def update_Lagrange(self):
        InvP_NFTMMM = self.xp.linalg.inv(self.P_NFTMMM)
        Inv_FTMMM = self.xp.linalg.inv(InvP_NFTMMM.sum(axis=0))
        if self.xp == "cp":
            InvP_NFTMMM = self.xp.array(InvP_NFTMMM)
            Inv_FTMMM = self.xp.array(Inv_FTMMM)
        #Id_FTMM =  self.xp.einsum("nftmlk,ftmlk->ftml",InvP_NFTMMM,Inv_FTMMM)
        # Id_FTMM = self.xp.einsum("ft, ml -> ftml",self.xp.eye(self.n_freq, self.n_time)
        #                         , self.xp.eye(self.n_mic))
        #Id_FTMM = self.xp.ones((self.n_freq, self.n_time, self.n_mic, self.n_mic))
        self.La_FTMM += self.xp.einsum("ftmlk, ftmk -> ftml",Inv_FTMMM ,
                         (self.W_NFTMM.sum(axis=0) - self.Id_NFTMMM[0,:,:,0]))
        del InvP_NFTMMM, Inv_FTMMM

    def update_W(self):
        Cste = float(4 * self.xp.pi / self.P)  # integration constant
        #WTh_NFTMP = (self.W_NFTMM[..., None].conj() * self.Theta_FPM.transpose(0, 2, 1)[None, :, None, None]).sum(axis=-2)
        
        WTh_NFTMP = self.xp.einsum("nftml, fpl -> nftmp",self.W_NFTMM.conj(),self.Theta_FPM)

        # N F T M "M" P -> N F T "M" M
        R_NFTMM = Cste * self.xp.einsum("fpl, fpm, nftmp, nft, nfp -> nftml",self.Theta_FPM, self.Theta_FPM.conj(), 
                self.xp.abs(self.Theta_FPM.transpose(0, 2, 1)[None, :, None, :]
                - WTh_NFTMP + self.eps) ** (self.alpha - 2.)  ,
                self.lambda_NFT , self.SM_NFP)

        InvP_NFTMMM = self.xp.linalg.inv(self.P_NFTMMM)
        self.W_NFTMM = self.xp.einsum("nftmlk, nftmk -> nftml",InvP_NFTMMM , (R_NFTMM - self.La_FTMM[None,:]))
        
        del Cste, WTh_NFTMP, R_NFTMM, InvP_NFTMMM

    def E_Step(self):
        del self.Psi_FPQ, self.Xi_FTP, self.W_NFK, self.H_NKT, self.Y_FTP
        # gc.collect()
        # mempool = self.xp.get_default_memory_pool()
        # mempool.free_all_blocks()

        # Init variables
        self.W_NFTMM = self.xp.ones((self.n_source, self.n_freq, self.n_time, self.n_mic, self.n_mic)).astype(self.xp.complex64)
        self.W_NFTMM *= (self.xp.eye(self.n_mic)/self.n_source)[None, None, None]
        self.La_FTMM = self.xp.zeros((self.n_freq, self.n_time, self.n_mic, self.n_mic)).astype(self.xp.complex64)
        #filenameP = path.join(mkdtemp(), 'P_NFTMMM.dat')     
        # self.P_NFTMMM = np.memmap(filenameP, dtype='complex64', mode='w+', shape=(self.n_source,self.n_freq,self.n_time,self.n_mic,self.n_mic,self.n_mic))
        self.P_NFTMMM = self.rand_s.rand(self.n_source, self.n_freq, self.n_time, self.n_mic, self.n_mic, self.n_mic).astype(self.xp.complex64)
        self.ThTh_FMMP = self.xp.einsum("fpm,fpl->fmlp",self.Theta_FPM , self.Theta_FPM.conj())
        self.Id_NFTMMM = self.xp.ones((self.n_source, self.n_freq, self.n_time,
                                  self.n_mic, self.n_mic, self.n_mic)) *\
                    self.xp.eye(self.n_mic)[None, None, None, None]
        cov_array = []
        for it in range(self.nE_it):
            print("filtrage iteration = {}".format(it))
            self.update_P()
            # ipdb.set_trace()
            self.P_NFTMMM += 1e-3 * self.xp.trace(self.P_NFTMMM) * self.Id_NFTMMM
            self.P_NFTMMM[self.xp.isnan(self.P_NFTMMM)] = self.eps
            self.update_Lagrange()
            self.La_FTMM[self.xp.isnan(self.La_FTMM)] = self.eps
            self.update_W()
            self.W_NFTMM[self.xp.isnan(self.W_NFTMM)] = self.eps
            if self.save_cov and (it > 10) and(it % self.interval_save_parameter == 0):
                self.cov_res = self.calculate_covariation()
                #ipdb.set_trace()
                cov_array.append(self.cov_res)
                for n in range(self.n_source):
                    #cov_array.append(self.cov_res)
                    #ipdb.set_trace()
                    plt.plot(self.interval_save_parameter * self.convert_to_NumpyArray(self.xp.arange(len(cov_array))),self.convert_to_NumpyArray(cov_array))
                    plt.show()
        #ipdb.set_trace()
        self.Y_NFTM = self.xp.einsum("nftml, ftl -> nftm", self.W_NFTMM.conj() , self.X_FTM)

    def calculate_log_likelihood(self):
        tmp_FTP = (self.a_4[0] * (self.X_FTP ** 4) / ((self.Y_FTP + self.eps) ** ((4 + 2 * self.n_mic) / self.alpha)) +\
                   self.a_4[1] * (self.X_FTP ** 2) / ((self.Y_FTP + self.eps) ** ((2 + 2 * self.n_mic) / self.alpha)) +\
                   self.a_4[2] * 1. / ((self.Y_FTP + self.eps) ** ((2 * self.n_mic) / self.alpha))) *\
            self.xp.exp(-self.a_4[3] * (self.X_FTP ** 2) / ((self.Y_FTP + self.eps) ** (2. / self.alpha)))

        ll_value = (self.xp.log(tmp_FTP.sum(axis=-1) + self.eps)).sum()
        return self.convert_to_NumpyArray(ll_value)

    def calculate_covariation(self):
        
        WTh_NFTMP = self.xp.einsum("nftml, fpl -> nftmp",self.W_NFTMM.conj(),
                     self.Theta_FPM)
                     
        tmp1 = self.xp.einsum("nftmp, nft, nfp -> n", 
                self.xp.abs(self.Theta_FPM.transpose(0, 2, 1)[None, :, None, :] - WTh_NFTMP) ** self.alpha
                - self.xp.abs(WTh_NFTMP)** self.alpha , self.lambda_NFT , self.SM_NFP)
        
        tmp2 = self.xp.einsum("nftmp, nft, nfp -> n",
                self.xp.abs(WTh_NFTMP) ** self.alpha , self.lambda_NFT , self.SM_NFP )
        
        cov = tmp1 + tmp2 
        
        return self.convert_to_NumpyArray(cov.mean())

    def make_filename_suffix(self):
        self.filename_suffix = "M={}-S={}-K={}-it={}".format(self.n_mic, self.n_source, self.n_basis, self.n_iteration)
        if self.oracle == True:
            self.filename_suffix = "oracle" + self.filename_suffix
        if self.init_parameter == True:
            self.filename_suffix = self.filename_suffix + '-init'

        if hasattr(self, "file_id"):
            self.filename_suffix += "-ID={}".format(self.file_id)
        return self.filename_suffix

    def convert_to_NumpyArray(self, data):
        if self.xp == np:
            return data
        else:
            return self.xp.asnumpy(data)

    def solve(self, n_iteration=100, save_likelihood=False,
              save_cov=False,
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
        self.interval_save_parameter = interval_save_parameter
        self.save_cov = save_cov
        self.n_iteration = n_iteration
        self.save_path = save_path
        self.init_Theta()
        self.init_Psi()
        self.init_SM()
        self.init_WH()
        # self.ref = 1 / self.n_mic * self.xp.einsum("nftm -> nft",self.X_NFTM)
        self.X_FTM = self.xp.einsum('nftm -> ftm',self.X_NFTM)
        # self.X_FTM /= self.xp.max(self.X_FTM, axis=0)
        self.init_auxfunc()
        self.make_filename_suffix()

        ll_array = []
        f_, ax = plt.subplots(2, 2)
        self.end = -1
        for it in range(self.n_iteration):
            # ipdb.set_trace()
            ax[0, 0].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.lambda_NFT[0])**2)), origin="lower", label='lambda')
            ax[0, 1].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.lambda_NFT[1])**2)), origin="lower", label='lambda')
            ax[1, 0].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.SM_NFP[0])**2)), origin="lower",label='gamma')
            ax[1, 1].imshow(self.convert_to_NumpyArray(self.xp.log(self.xp.abs(self.SM_NFP[1])**2)), origin="lower",label='gamma')
            plt.show()
            if self.oracle == True:
                self.update_WH()
                self.normalize()
                # print(self.xp.sum(self.xp.isnan(self.Xi_FTP)))
            if self.oracle == False:
                self.update_SM()
                # print(self.xp.sum(self.xp.isnan(self.SM_NFP)))
            if self.update_psi:
                self.update_Psi()
            #ipdb.set_trace()
            if save_likelihood and ((it+1) % self.interval_save_parameter == 0) and ((it+1) != self.n_iteration):
                ll_res = self.calculate_log_likelihood()
                ll_array.append(ll_res)
                plt.plot(self.interval_save_parameter * self.convert_to_NumpyArray(self.xp.arange(len(ll_array))),ll_array)
                plt.show()
                #plt.savefig(save_path + "test{}.png".format(it))
            self.end += 1 
        g_model = open( save_path + "{}-{}".format(self.method_name, self.filename_suffix) + "SM_update.pkl", 'wb')
        SM_ = self.convert_to_NumpyArray(self.SM_NFP)
        pkl.dump(SM_, g_model)
        del g_model, SM_

        if save_likelihood and (self.end+1 == self.n_iteration):
            ll_res = self.calculate_log_likelihood()
            ll_array.append(ll_res)
            plt.plot(ll_array)
            plt.show()
            f_model = open( save_path + "{}-{}".format(self.method_name, self.filename_suffix) + 'log_vraissemblance-' + "-N=1.pkl", 'wb')
            pkl.dump(ll_array, f_model)
            del f_model
            del ll_res, ll_array
            # plt.savefig(save_path + "{}-likelihood-interval={}-{}.png".format(self.method_name, interval_save_parameter, self.filename_suffix))
            # pkl.dump(ll_array, open(save_path + "{}-likelihood-interval={}-{}.pic".format(self.method_name, interval_save_parameter, self.filename_suffix), "wb"))
        if save_wav and ((self.end+1) == self.n_iteration):
            self.E_Step()
            #self.save_separated_signal(save_path + "{}-{}".format(self.method_name, self.filename_suffix))

        f_model = open(save_path + "{}-{}".format(self.method_name, self.filename_suffix) + "-N=1.pkl", 'wb')
        self.Y_NFTM = self.convert_to_NumpyArray(self.Y_NFTM)
        pkl.dump(self.Y_NFTM, f_model)
        del f_model

        if save_parameter:
            with h5py.File(save_path + "spatial-measure-{}-{}.hdf5".format(self.method_name, self.filename_suffix), 'w') as file_obj:
                file_obj.create_dataset(
                    'SM_NFP',
                    data=self.convert_to_NumpyArray(self.SM_NFP).astype(np.float32),
                    dtype=np.float32,
                    compression='lzf')

    def save_separated_signal(self, save_fileName="sample.wav"):
        del self.P_NFTMMM, self.La_FTMM
        self.Y_NFTM = self.convert_to_NumpyArray(self.Y_NFTM)
        hop_length = int((self.n_freq - 1) / 2)
        for n in range(self.n_source):
            for m in range(self.n_mic):
                tmp = sc.signal.istft(self.Y_NFTM[n, :, :, m]) #,hop_length=hop_length
                if n == 0 and m == 0:
                    separated_signal = np.zeros([self.n_source, len(tmp), self.n_mic])
                separated_signal[n, :, m] = tmp
        separated_signal /= np.abs(separated_signal).max() * 1.2
        print("taille signaux separes",separated_signal.shape)
        
        SDR, SIR, SAR, PERM = [], [], [], []
        
        # for i in range self.n_source:
        #     sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(ref.sum(axis=1), separated_signal.sum(axis=-1))
        #     SDR.append(sdr)
        #     SIR.append(sir)
        #     SAR.append(sar)
        #     PERM.append(perm)
        # plt.plot()
        
        for n in range(self.n_source):
            sf.write(save_fileName + "-N={}.wav".format(n), separated_signal[n,:,:], 44100)
