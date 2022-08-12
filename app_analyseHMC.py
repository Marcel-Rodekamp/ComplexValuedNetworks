import numpy as np
import torch

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import itertools as it

import h5py as h5

from lib_tensorSpecs import torchTensorArgs

# The Layers are implemented using torch.nn.Module:
# Implementation the Paired (Random) Coupling Layer (PRCL)
from lib_layer import PRCL
# Implementation of the Affine Coupling
from lib_layer import AffineCoupling
# Implementation of a linear (fully connected) layer for 2 dimensional 
# input configuration
from lib_layer import LinearTransformation

# The Layers are set up to calculate the log det on the fly. 
# This requires a special implementation of the sequential container
from lib_layer import Sequential

from lib_activations import complexRelu, zLogz, fractions

from lib_2SiteModel import Hubbard2SiteModelIsleIsleAction

from lib_loss import StatisticalPowerLoss,MinimizeImaginaryPartLoss

from lib_train import train

from lib_trainAnalysisHMC import plot_loss, plot_actionStatistics2, plot_fieldStatistics2, plot_loss_eq,  plot_loss_eq2, decodeHyperparams, plot_action_evolution, plot_stat_power, plot_correlators, pathname



Nt = 16
beta = 4
U = 4
mu = 3
tangentPlaneOffset = -4.99933002e-01
Nx = 2

# lattice spacing
delta= beta/Nt

# Make all variables unit less
U  = delta * U
mu = delta * mu

 
# hopping matrix (particles)
# exp( \kappa + C^p )
expKappa_p = torch.zeros((Nx, Nx),**torchTensorArgs)
expKappa_p[0, 0] = np.cosh(delta) * np.exp(mu)
expKappa_p[0, 1] = np.sinh(delta) * np.exp(mu)
expKappa_p[1, 0] = np.sinh(delta) * np.exp(mu)
expKappa_p[1, 1] = np.cosh(delta) * np.exp(mu)

# hopping matrix (holes)
# exp( \kappa + C^h )
expKappa_h = torch.zeros((Nx, Nx),**torchTensorArgs)
expKappa_h[0, 0] =  np.cosh(delta) * np.exp(-mu)
expKappa_h[0, 1] = -np.sinh(delta) * np.exp(-mu)
expKappa_h[1, 0] = -np.sinh(delta) * np.exp(-mu)
expKappa_h[1, 1] =  np.cosh(delta) * np.exp(-mu)

def M(phi, species):

    Nt,Nx = phi.shape
    
    M = torch.zeros((Nt, Nx, Nt, Nx), **torchTensorArgs)
    
    if species == 1:
        expKappa = expKappa_p
    elif species == -1:
        expKappa = expKappa_h
    else:
        return ValueError(f"Fermion Matrix got wrong species argument. Must be +/- 1 but is {species}.")
    
    
    expphi = torch.exp(species*1.j*phi)
    
    ts = torch.arange(0,Nt-1)
    M[ts,:,ts,:] = torch.eye(Nx,**torchTensorArgs)
    
    M[ts+1, :, ts, :] = -expKappa[:, :] * expphi[ts,None, :]

    M[Nt - 1, :, Nt - 1, :] = torch.eye(Nx,**torchTensorArgs)
  
    M[0, :, Nt-1, :] = expKappa[:, :] * expphi[Nt-1,None, :]

    return M

def bootstrap(data, weights, Nbst = 100):
    f"""..."""
    Nconf,Nt,Nx,_ = data.shape
    correl = np.zeros( shape = (Nbst,Nt,Nx,Nx) )
    #error = np.zeros( shape = (Nbst,Nt,Nx,Nx) )
    
    for k in range(Nbst):
        indices = np.random.randint(0,Nconf, size=Nconf)
        matrix_sample = data[indices,:,:,:]
        weights_sample = weights[indices]
        numerator_sample = matrix_sample * weights_sample[:,None,None,None]
        numerator_mean = numerator_sample.mean(axis=0).detach().numpy()
        #numerator_error = numerator_sample.std(axis=0).detach().numpy()
        denominator_mean = weights_sample.mean(axis=0).detach().numpy()
        
        correl[k] = numerator_mean/denominator_mean
        #error[k] = numerator_error/denominator_mean

    return correl.mean(axis=0), correl.std(axis = 0)


def StatPower(x, S_eff, Nbst):
    N = x.size
    result = torch.zeros(N)
    error = torch.zeros(N)
        
    for i in range(N):
        
        mean = np.zeros(Nbst)
        
        for k in range(Nbst):
            indices = np.random.randint(0,i+1, size = i+1)
            ordered_configs = S_eff.imag[:(i+1)]
            sample = torch.exp(-1j*ordered_configs[indices])
           
            mean[k] = sample.mean(axis=0).abs()
            
        result[i] = mean.mean()
        error[i]  = mean.std()
        
        
    return result, error

def correlator(phis, actions):
    r"""
        \params:
            - phi, torch tensor, shape = (Nt,Nx)
    
        This function computes the correlator
    
        C_{x,y}(t) = <p^\dagger_x p_y> = <Mp^{-1}_{t,x,t,y}>
    """
    # Get the size of the system
    Nconf,_,_ = phis.size()
    
    # Reweight with the action
    weights = (-1j * actions).exp()
    
    # Construct the fermion matrix
    Mp = torch.zeros((Nconf,Nt,Nx,Nt,Nx),**torchTensorArgs)
    for n in range(Nconf):
        Mp[n,:,:,:,:] = M(phis[n,:,:],+1)
    
    # Invert the fermion matrix
    MpInv = torch.linalg.inv(Mp.reshape(Nconf,Nt*Nx,Nt*Nx)).reshape(Nconf,Nt,Nx,Nt,Nx)
    
    # Compute the correlator
    
    data = MpInv[:,:,:,0,:]
    
    
    Cxy_est, Cxy_err = bootstrap(data, weights, Nbst = 100) 

    return Cxy_est,Cxy_err 


if __name__ == "__main__":
    
    
    # load the stuff
    
    name0 = "Configs_,Nt_16initEpsilon_0.001,trainEpsilon_1e-05,trainBatchSize_10000,trainMiniBatchSize_10000,learningRate_1e-07, numPRCLLayers_1,numInternalLayers_8, activation_Softsign(), lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>.h5"
    
    name1 = "Configs_Nt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_1e-07 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1200.h5"
    
    name2 = "Configs_Nt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_1000LR_1e-06 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1600.h5"
    
    name3 = "Configs_Nt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_5e-06 numPRCLLayers_1numIntLayers_2activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1000.h5"
    
    name4 = "Configs_Nt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_1e-06 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1200.h5"
    
    name5 = "Configs_Nt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_1000LR_1e-07 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1400.h5"
    
    #names = [name0, name1, name2, name3, name4, name5]
    names = [name0]
    
    hyps0 = {
          "numPRCLLayers": 1,
          "numInternalLayers": 8,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 10000,
          "lossFct": MinimizeImaginaryPartLoss,
          "learningRate": 1e-07,
          "Nt": 16
        }
    
    hyps1 = {
          "numPRCLLayers": 1,
          "numInternalLayers": 8,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 5000,
          "lossFct": MinimizeImaginaryPartLoss,
          "learningRate": 1e-07,
          "Nt": 16
        }
    
    hyps2 = {
          "numPRCLLayers": 1,
          "numInternalLayers": 8,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 1000,
          "lossFct": MinimizeImaginaryPartLoss,
          "learningRate": 1e-06,
          "Nt": 16
        }
    
    hyps3 = {
          "numPRCLLayers": 1,
          "numInternalLayers": 2,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 5000,
          "lossFct": MinimizeImaginaryPartLoss,
          "learningRate": 5e-06,
          "Nt": 16
        }
        
    hyps4 = {
          "numPRCLLayers": 1,
          "numInternalLayers": 8,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 5000,
          "lossFct": MinimizeImaginaryPartLoss,
          "learningRate": 1e-06,
          "Nt": 16
        }
    
    hyps5 = {
          "numPRCLLayers": 1,
          "numInternalLayers": 8,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 1000,
          "lossFct": MinimizeImaginaryPartLoss,
          "learningRate": 1e-07,
          "Nt": 16
        }
    
    #hyps = [hyps0, hyps1, hyps2, hyps3, hyps4, hyps5]
    hyps = [hyps0]
    Nconf = 10000
    burnIn = 1000
    Nt = 16
    
    for hyperparameters, filename in zip(hyps, names):
        
        
        with h5.File(filename,'r') as h5f:
            
            configsM = torch.from_numpy( h5f["configsM"] [()] )
            configsR = torch.from_numpy( h5f["configsR"] [()] )
            S = torch.from_numpy( h5f["S"] [()] )
            S_full = torch.from_numpy( h5f["S_full"] [()] )
            logDetJ_NN = torch.from_numpy( h5f["logDetJ_NN"] [()] )
            logDetJ_NN_full = torch.from_numpy( h5f["logDetJ_NN_full"] [()] )
            accRate = h5f["accRate"] [()] 
            S_eff = torch.from_numpy( h5f["S_eff"] [()] )
            
            
         
    
        # calculate the statistical power

        x=np.arange(0,Nconf, 100)

        result, error = StatPower(x, S_full, 100)


        # plots

        plot_action_evolution(S_full, S, burnIn, Nconf, hyperparameters)
        print(f"Acceptance Rate = {accRate}")

        plot_stat_power(x,result.detach().numpy(), error.detach().numpy(), hyperparameters)

        # calculate the correlators

        Cxy_estM,Cxy_errM = correlator(phis=configsM, actions = S_eff)
        

        print("Cxy_estM shape is: ", Cxy_estM.shape)
        print("Cxy_estM:")
        print(Cxy_estM)
        print("Cxy_errM shape is: ", Cxy_errM.shape)
        print("Cxy_errM:")
        print(Cxy_errM)


        t = np.arange(0,16)

        plot_correlators(t,Cxy_estM, Cxy_errM, hyperparameters)

        print("S_eff.imag is ")
        print(S_eff.imag)
