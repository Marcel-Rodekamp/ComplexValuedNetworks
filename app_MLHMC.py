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

from lib_trainAnalysisHMC import plot_loss, plot_actionStatistics2, plot_fieldStatistics2, plot_loss_eq,  plot_loss_eq2, decodeHyperparams, plot_action_evolution, plot_stat_power, plot_correlators, pathnameh


try:
    import isle 
    import isle.action
except:
    print("No isle support found.")  
    
    
# define the parameters of the Hubbard Model as global variables (awful thing to do, I know)

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

def coupling_factory(numInternalLayers, internalLayer_factory, activation_factory, initEpsilon, **internalLayerKwargs):
    r"""
        \param:
            - numInternalLayers 
            - internalLayer_factory 
            - activation_factory 
            - initEpsilon 
            - internalLayerKwargs
        This factory generates an affine coupling layer (see lib_layer.py)
        using numInternalLayers of torch.nn.Modules created by internalLayer_factory 
        an seperated by activation_factory. 
        Depending on the number of numInternalLayers this generates a coupling
        of the form 
        ```
        AffineCoupling:
            m = Sequential([ m_layer1, activation, m_layer2, activation, ..., m_layerN ]),
            a = Sequential([ a_layer1, activation, a_layer2, activation, ..., a_layerN ]),
        ```
    """
    
    def generate_parameters():
        r"""
            This generates the list of internal layers used for the networks 
            called a,m in the affine coupling
            It initializes the parameters of these layes using a uniform 
            distribution between -initEpsilon and +initEpsilon
        """
        layerList = []
        for _ in range(numInternalLayers-1):
            # create internal layers using the factory
            layer = internalLayer_factory(**internalLayerKwargs)
            
            # initialize the parameters using a uniform distribution
            for param in layer.parameters():
                torch.nn.init.uniform_(param,-initEpsilon,initEpsilon)

            layerList.append(layer)
            
           # append a activation function
            layerList.append(activation_factory())

        # generate and initialize a last layer which is not followed by 
        # an activation function
        layer = internalLayer_factory(**internalLayerKwargs)
        for param in layer.parameters():
            torch.nn.init.uniform_(param,-initEpsilon,initEpsilon)
        
        layerList.append(layer)

        # put everything in a torch sequential container to apply 
        # the layers after each other
        return torch.nn.Sequential(*layerList)
    
    return AffineCoupling(
        m = generate_parameters(),
        a = generate_parameters()  
    )

def NN_factory(numPRCLLayers,HM,coupling_factory,**couplingKwargs):
    r"""
        \param:
            - numPRCLLayers:   int, number of PRCL layers stored in the module
            - HM:              Hubbard2SiteModelIsleIsleAction instance carrying information to set up the PRCL layers  
            - coupling_factory: callable, generating a coupling for the PRCL layer (both couplings in PRCL are generated in the same way) 
            - couplingKwargs: keyworded arguments for the coupling_factory
        This factory function creates a `numPRCLLayers` PRCL layer using 
        coupling(1,2) created by the coupling_factory. The resulting torch 
        module could look something like
        ```
        Sequential:
            PRCL(1):
                coupling1 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                ),
                coupling2 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                )
            PRCL(2):
                coupling1 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                ),
                coupling2 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                )
        ```
        The form of the coupling(1,2) varies depending on the coupling 
        factory.
    """

    # Store all PRCL layers in one list, later passed to the Sequential container
    NN = []

    for _ in range(numPRCLLayers):
        # Create a PRCL layer with couplings defined by the coupling factories
        NN.append(
            PRCL(
                HM.Nt,HM.Nx,
                coupling1 = coupling_factory(**couplingKwargs),
                coupling2 = coupling_factory(**couplingKwargs)
            )
        )

    return Sequential(NN)

class Equivariance(torch.nn.Module):
    def __init__(self,NN):
        
        # initialise the class
        super(Equivariance,self).__init__()
        
        # add the NN as a module
        self.add_module("NN",NN)

    def forward(self,phi):
        r"""
            \param: phi: batch of configurations
            The forward pass will implement the TL layer, using the NN
        """
        
        # get the dimensions of phi
        Nconf,_,_ = phi.shape
        
        # get the time slice t0 where each config has a minimum value along x=0
        t = phi.abs().argmin(dim=1)
        
        T0 = t[:,0]
       
        # loop through all the configurations in the batch and roll them
        # (the 'for' loop is needed because 'shifts' can only take int as argument)
        for i in range(Nconf):
            config = phi[i,...]
            if t[i][0].item() > t[i][1].item():
                T0[i] = t[i][1]
                config.flip(1)
                            
            # translate the whole configuration in time by t0
            config = torch.roll(config, shifts = T0[i].item(), dims=1)
            
            # assemble back into a batch of configs
            phi[i,...] = config
            
        # apply the NN
        NNphi, logDet = self.NN(phi)
        
        # invert the time translation, rolling back with -t0
        for i in range(Nconf):
            NNphi[i,...] = torch.roll(NNphi[i], shifts = -T0[i].item(), dims=0)
            if t[i][0].item() > t[i][1].item():
                NNphi[i,...].flip(1)
        
        # this whole class is supposed to be a new NN, so again we return the resulting configurations and the logDet
        # the logDet is unchanged by the time translation, so it is just the logDet returned by the initial NN
        return NNphi, logDet


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

def logdetM(phi,species):    
    
    return torch.log(torch.linalg.det(M(phi,species).reshape(Nt*Nx,Nt*Nx)))


# the action of the system
def action(mu, phi):
   
    return (phi*phi).sum()/(2*U) - logdetM(mu, phi,+1) - logdetM(mu, phi,-1) 

# The force term from log det M^p
def TrMinvM(phi,species):
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
            - species: +1 for particles, -1 for holes
        
        \frac{d \log{\det{ M(\phi) }}}{ d\phi } = \Tr{ M^{-1}(\phi) \cdot \frac{dM(\phi)}{d\phi} }
        
        where 
        
        \frac{dM(\phi)_{t',x'; t,x}} }{d\phi_{t,x}} = -i*s* exp(s*(\Kappa+\mu))_{x',x} exp(i*s*\phi_{t,x}) \delta_{t',t+1} 
    """
    # Tr = d logdetM(\phi)/d\phi_{t,x}
    Tr = torch.zeros((Nt,Nx),**torchTensorArgs)
    
    # determine the expression of exp(s*Kappa) from the species 
    if species == 1:
        expKappa = expKappa_p
    elif species == -1:
        expKappa = expKappa_h
    else:
        return ValueError(f"Force got wrong species argument. Must be +/- 1 but is {species}.")
    
    expphi = torch.exp(species*1j*phi)
    
    # again reshape M for torch to understand it as a matrix
    Minv = torch.linalg.inv(
        M(phi,species).reshape(Nt * Nx, Nt * Nx)
    ).reshape(Nt, Nx, Nt, Nx)
    
    ts = torch.arange(0,Nt-1)
    
    #bulk time slices
    #for t in range(Nt-1): 
        #for x in range(Nx):
            # temp_{t,x} = sum_{t'} M^{-1}_{t,x; t',x'} exp(s*(\Kappa+\mu))_{x',x} \delta_{t',t+1}
            # This is the matrix multiplication with y=x'
            # The sum over t' is resolved with the delta_{t',t+1}
        
    temp = torch.diagonal(torch.tensordot(Minv[ts,:,ts+1,:],expKappa[:,:],dims=1), dim1 = 1, dim2 = -1)
        
        #print(temp)    
            # (TrMinvM)_{t,x} = -i * s * temp_{t,x} * exp(i*s*\phi_{t,x})
    Tr[ts, :] = -1.j * species * temp[:] * expphi[ts, :]
    
    # boundary time slice
    # term t = Nt=0, t' = Nt=-1 
    #for x in range(Nx):
        # temp_{Nt-1,x} = M^{-1}_{0,x; t'=0,x'} exp(s*(\Kappa+\mu))_{x',x} \delta_{t'=0,t+1}tmp_{t,x} 
        #               = M^{-1}_{t,x,t+1} @ exp(Kappa) 
    temp = torch.diag(torch.tensordot(Minv[Nt-1,:,0,:],expKappa[:,:],dims=1))
    # (TrMinvM)_{Nt-1,x} = i * s * tmp_{t,x} * exp(i*s*phi_{t,x})
    Tr[Nt - 1, :] = 1.j * species * temp * expphi[Nt-1, :]

    return Tr

def TrMinvM_ph(phi):
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
        
         Compute the fermionic part of the force (see TrMinvM) for both species
    """
    return TrMinvM(phi,+1) + TrMinvM(phi,-1)
def force(phi):
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
            
        F(phi) = - dS(phi)/dphi = - d/dphi ( 1/2U phi^2 - log det M^p - log det M^h )
                                = - [ 1/U phi - Tr( M^{p,-1} dM^p/dphi + M^{h,-1} dM^h/dphi ) ]
    """
    
    return -(phi/U - TrMinvM_ph(phi))


def leapfrog(phi0,pi0,trajLength,Nmd,direction = 1):
    r"""
        \params:
            - phi0: torch.tensor(Nt,Nx), start configuration 
            - pi0 : torch.tensor(Nt,Nx), momentum field (complex dtype but with vanishing imaginary part)
            - trajLength: float, trajectory length i.e. time to which the eom are integrated
            - Nmd : int, number of molecular dynamics steps 
            - direction: int, direction (+/-) in which the algorithm should integrade
            
        This function integrades the Hamilton Equation of Motion (see hamiltonian and force) using the 
        leapfrog algorithm.
        
        This algorithm is reversible up to numerical precision and energy preserving up to order epsilon^2 where
        epsilon = trajLength/Nmd
    """
    # deepcopy the fields to not change the original torch tensors
    phi = phi0.clone()
    pi = pi0.clone()
    
    # compute the step size and specify the integration direction 
    stepSize = direction*trajLength/Nmd 
    
    # first half step
    # x_{1/2} = x_{0} + 0.5*\Delta t * p_{0}
    phi+= 0.5 * stepSize * pi
    
    # a bunch of full steps
    # p_{t+1} = p_{t} + \Delta t Re(dS/dphi)
    # x_{t+3/2} = x_{t+1/2} + \Delta t * p_{t+1}
    
    Nt,Nx = phi.shape 
    
    for _ in range(Nmd-1):
        pi += stepSize * force(phi)    
        phi+= stepSize * pi
    # final half step
    # p_{Nmd-1} = p_{Nmd-2} + \Delta t Re(dS/dphi)
    # x_{Nmd-1} = x_{Nmd-3/2} + 0.5 \Delta t * p_{Nmd-1}
    
    
    pi += stepSize * force(phi)
    phi+= 0.5 * stepSize * pi
    
    return phi,pi

def MLHMC(phiR0, trajLength, Nmd, Nconf, burnIn, transformation, thermalization):
    r"""
        \params:
            - phi0:           torch.Tensor(Nt,Nx), start element of the markov chain
            - trajLength:     float, Molecular Dynamics integration length, see leapfrog
            - Nmd:            int,  number of integration steps, see leapfrog
            - Nconf:          int, number of configurations (after burn in and without thermalization)
            - burnIn:         int, number of configurations to be discarded at the beginning of the algorithm (thermalization from phi0)
            - transformation: torch.nn.Module, network trained to move the configuration to a sign problem reduced manifold
            - thermalization: int, number of configurations to be discarded between two markov elements (reducing the autocorrelation)
    
        Implementation of a HMC algorithm, augmented with machine learning to alleviate the sign problem.
    """
    
    def hmc_step(phiM_n, phiR_n, logDetJ_NN_n):
        # sample a momentum field pi ~ N(0,1)
        pi = torch.normal(
            mean = torch.zeros((HM.Nt,HM.Nx),**torchTensorArgs).real, 
            std  = torch.ones((HM.Nt,HM.Nx),**torchTensorArgs).real
        ) + 0.j
        
        # Compute initial hamiltonian value
        S0 = HM.calculate_batch_action(phiM_n)
        H0 = S0 + 0.5*(pi*pi).sum() - logDetJ_NN_n
        
        # Integrate Hamiltons EoM to generate a proposal
        # This integration needs to be done on the real plane!
        
        phiR,pi = leapfrog(
            phi0 = phiR_n,
            pi0  = pi,
            trajLength = trajLength,
            Nmd    = Nmd,
        )
        
        # transform to the more optimal manifold
        batchphiR = phiR.unsqueeze(dim=0)
        batchphiM, logDetJ_NN = transformation(batchphiR)
        phiM = batchphiM.squeeze(dim=0)
        
        # Compute final hamiltonian value
        S1 = HM.calculate_batch_action(phiM)
        H1 = S1 + 0.5*(pi*pi).sum() - logDetJ_NN
        
        # WHY DOES THE MOMENTUM FIELD REMAIN UNCHANGED?
        
        #print("the probability to accept is ", torch.exp( -(H1-H0).real ))

        
        # accept reject
        if torch.rand(1).item() <= torch.exp( -(H1-H0).real ).item():
            return phiM,phiR,logDetJ_NN, S1, 1
        else:
            return phiM_n,phiR_n,logDetJ_NN_n, S0, 0
    
    # create a list of configurations with length Nconf
    # manifold configs
    markovChainM = torch.zeros( (Nconf,HM.Nt,HM.Nx), **torchTensorArgs )
    # real plane configs
    markovChainR = torch.zeros( (Nconf,HM.Nt,HM.Nx), **torchTensorArgs )
    
    # create a list of action values
    action_full = torch.zeros( (Nconf+burnIn), **torchTensorArgs )
    action_markovChain = torch.zeros( (Nconf), **torchTensorArgs )
    
    # create a list of weights i.e. logDet J_NN
    logDetJ_NN_full = torch.zeros( (Nconf+burnIn), **torchTensorArgs )
    logDetJ_NN_markovChain = torch.zeros( (Nconf), **torchTensorArgs )
    
    # perform burnIn calculations
    phiR_n = phiR0.clone()
    batchphiR_n = phiR_n.unsqueeze(dim=0)
    batchphiM_n, logDetJ_NN_n = transformation(batchphiR_n)
    phiM_n = batchphiM_n.squeeze(dim=0)
    for n in range(burnIn):
        phiM_n,phiR_n,logDetJ_NN_n,S_n,_ = hmc_step(phiM_n,phiR_n,logDetJ_NN_n)
        action_full[n] = S_n
        #print(logDetJ_NN_n)
        logDetJ_NN_full[n] = logDetJ_NN_n

    # store starting point of HMC
    markovChainM[0,:,:] = phiM_n
    markovChainR[0,:,:] = phiR_n
    action_markovChain[0] = action_full[burnIn-1]
    logDetJ_NN_markovChain[0] = logDetJ_NN_full[burnIn-1]
    
    acceptenceRate = 0
    
    logDetJ_NN_n = logDetJ_NN_markovChain[0] 
            
    # perform markov chain calculation
    for n in range(Nconf-1):
        markovChainM[n+1],markovChainR[n+1],logDetJ_NN_markovChain[n+1],action_markovChain[n],acceptence = \
            hmc_step(phiM_n,phiR_n,logDetJ_NN_n)
        
        action_full[n+burnIn] = action_markovChain[n]
        logDetJ_NN_full[n+burnIn] = logDetJ_NN_markovChain[n+1]
        
            
        acceptenceRate += acceptence
        
        phiM_n = markovChainM[n+1]
        phiR_n = markovChainR[n+1]
        logDetJ_NN_n = logDetJ_NN_markovChain[n+1]
        if n != Nconf-2:
            for _ in range(thermalization):
                phiM_n,phiR_n,logDetJ_NN_n,S_n,_ = hmc_step(phiM_n,phiR_n,logDetJ_NN_n)
        
    return {
        "configsM": markovChainM,
        "configsR": markovChainR,
        "S": action_markovChain,
        "S_full": action_full,
        "log Det J_NN": logDetJ_NN_markovChain,
        "log Det J_NN full": logDetJ_NN_full,
        "acceptance rate": acceptenceRate/Nconf,
        "S_eff": action_markovChain - logDetJ_NN_markovChain
    }




if __name__ == "__main__":
    
    # create the set of nets
    
    name0 = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_10000LR_1e-07 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1000.pt"
    
    name1 = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_1e-07 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1200.pt"
    
    name2 = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_1000LR_1e-06 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1600.pt"
    
    name3 = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_5e-06 numPRCLLayers_1numIntLayers_2activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1000.pt"
    
    name4 = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_1e-06 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1200.pt"
    
    name5 = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_1000LR_1e-07 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn_<class 'lib_loss.MinimizeImaginaryPartLoss'>epochs_1400.pt"
    
    names = [name1, name2, name3, name4, name5]
    
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
    
    hyps = [hyps1, hyps2, hyps3, hyps4, hyps5]
   
    
    # Load the Hubbard Model
    HM = Hubbard2SiteModelIsleIsleAction(
        Nt = Nt,
        beta = beta,
        U = U,
        mu = mu,
        tangentPlaneOffset = tangentPlaneOffset
    )

    # define the parameters of the HMC

    #trajectoryLength = 0.1
    #N_moleculardynamics = 10
    trajectoryLength = 0.1
    N_moleculardynamics = 10
    Nconf = 10000
    burnIn = 1000
    thermalization = 10
    mu = HM.mu

    # analyse each of the NN's that we saved
    
    for hyperparameters, filename in zip(hyps, names):
        
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate = \
            decodeHyperparams(hyperparameters)
        activation = torch.nn.Softsign
        # Initialise a new NN with the current hyperparameteres
        NN = NN_factory(
            numPRCLLayers = numPRCLLayers,
            HM = HM,
            coupling_factory = coupling_factory,
            # coupling Kwargs
            numInternalLayers = numInternalLayers, 
            internalLayer_factory = LinearTransformation, 
            activation_factory = activation, 
            initEpsilon = initEpsilon, 
            # Linear Transformation Args
            Nt = HM.Nt,
            Nx = HM.Nx//2
        )
        
        
        NN = Equivariance(NN)
        NN.load_state_dict(torch.load(filename))
        # generate test data: the first element of the Markov chain
        testData = torch.zeros((HM.Nt,HM.Nx),**torchTensorArgs).uniform_(-1e-5,1e-5).real + 0*1j
        
        # apply the HMC

        res = MLHMC(
            phiR0 = testData, 
            trajLength = trajectoryLength, 
            Nmd = N_moleculardynamics, 
            Nconf = Nconf, 
            burnIn = burnIn, 
            transformation = NN,
            thermalization = thermalization
        )
        
        
        configsM = res['configsM']
        configsR = res['configsR']
        S = res['S']
        S_full = res['S_full']
        logDetJ_NN = res['log Det J_NN']
        logDetJ_NN_full = res['log Det J_NN full']
        accRate = res['acceptance rate']
        S_eff = res['S_eff']
        
        # save this as a data frame
        
        
        
        filename = f"Configs_{pathnameh(hyperparameters)}"
        
        with h5.File(filename,'a') as h5f:
            h5f.create_dataset("configsM", data = configsM.detach().numpy())
            h5f.create_dataset("configsR", data = configsR.detach().numpy())
            h5f.create_dataset("S", data = S.detach().numpy())
            h5f.create_dataset("S_full", data = S_full.detach().numpy())
            h5f.create_dataset("logDetJ_NN", data = logDetJ_NN.detach().numpy())
            h5f.create_dataset("logDetJ_NN_full", data = logDetJ_NN_full.detach().numpy())
            h5f.create_dataset("accRate", data = accRate)
            h5f.create_dataset("S_eff", data = S_eff.detach().numpy())
        
        
