#!/usr/bin/env python
# coding: utf-8

# # Demonstration: Complex-Valued Affine Coupling Layer
# 
# This notebook is meant to show how the affine coupling layer - with complex-valued trainable parameters - works.
# To show a typical show case, the next lines of code implement the 2-site Hubbard model. 
# For more information about the methods referr to [our paper](https://arxiv.org/abs/2203.00390).


# Everything is set up to work in torch to make interfacing with the neural network easier
import torch

# plotting is done using matplotlib 
import matplotlib.pyplot as plt

# matplotlib works better with numpy.
import numpy as np

# For fitting wie simply use
from scipy.optimize import curve_fit

# Easier looping
import itertools as it


# The Layers are implemented using torch.nn.Module:
# Implementation the Paired (Random) Coupling Layer (PRCL)
from layer import PRCL
# Implementation of the Affine Coupling
from layer import AffineCoupling

# The Layers are set up to calculate the log det on the fly. 
# This requires a special implementation of the sequential container
from layer import Sequential



# the code is absolutely not optimized for GPU nor could the 
# 2 site problem fill a GPU thus it might be best to keep it 
# with device = 'cpu'.
torchTensorArgs = {
    "device": torch.device('cpu'),
    "dtype" : torch.cdouble
}


# ## Simulation parameters
# 
# The following cell defines the parameter of the Hubbard model. 

### Number of time slices
Nt   = 4

# Controle the continuum limit with
beta = 1

# On site interaction
U    = 1

# chemical potential
mu   = 1

# lattice spacing
delta= beta/Nt

# Put to 0 if not desired:
# tangent plane 
tpOffset = -6.97420945e-02 # Nt=4,beta=U=mu=1
# tpOffset = 0



# Don't change anything in this cell! 
# It provides named variables to increase readability of the code

# number of ions in the lattice
Nx = 2

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


def M(phi,species):
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
            - species: +1 for particles, -1 for holes
        
        M(phi)_{t',x'; t,x} = \delta_{t',t} \delta_{x',x} - exp(s*(\Kappa+\mu))_{x',x} exp(i*s*\Phi_{t,x}) \delta_{t',t+1} 
    """
    Nt,Nx = phi.shape
    
    M = torch.zeros((Nt, Nx, Nt, Nx), **torchTensorArgs)
    
    # determine the expression of exp(s*Kappa) from the species 
    if species == 1:
        expKappa = expKappa_p
    elif species == -1:
        expKappa = expKappa_h
    else:
        return ValueError(f"Fermion Matrix got wrong species argument. Must be +/- 1 but is {species}.")
    
    # precompute the exponential of the configuration
    
    expphi = torch.exp(species*1.j*phi)
    
    ts = torch.arange(0,Nt-1)
    M[ts,:,ts,:] = torch.eye(Nx,**torchTensorArgs)
    
    M[ts+1, :, ts, :] = -expKappa[:, :] * expphi[ts,None, :]
    # bulk time slices
    #for t in range(Nt - 1):
        # \delta_{t',t} \delta_{x',x}
       
    
    # \delta_{t',t} \delta_{x',x}
    M[Nt - 1, :, Nt - 1, :] = torch.eye(Nx,**torchTensorArgs)
    
    # bundary time slice 
    # term t' = Nt = 0,  t = Nt-1
        # exp(s*(\Kappa+\mu))_{x',x} exp(i*s*\Phi_{t,x}) \delta_{t',t+1}
    
    M[0, :, Nt-1, :] = expKappa[:, :] * expphi[Nt-1,None, :]

    return M



# to define the action we will need the log det of the matrices
def logdetM(phi,species):    
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
            - species: +1 for particles, -1 for holes
        
        \log \det [M(phi)_{t',x'; t,x}] 
    """
    # For torch to handle M as a matrix we can simply reshape the 4 rank tensor to a 2 rank tensor (i.e. matrix)
    return torch.log(torch.linalg.det(M(phi,species).reshape(Nt*Nx,Nt*Nx)))



# the action of the system
def action(phi):
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
            - species: +1 for particles, -1 for holes
        
        S(\phi) = \frac{1}{2U} \phi^T \phi - \log{\det{ M^p \cdot M^h }}   
    """
    return (phi*phi).sum()/(2*U) - logdetM(phi,+1) - logdetM(phi,-1) 


# generalized MD hamiltonian
def hamiltonian(phi,pi):
    r"""
        \params:
            - phi: torch.tensor(Nt,Nx), configuration
            - species: +1 for particles, -1 for holes
        
        H(\phi) = \frac{1}{2} \pi^T \pi + S(\phi)   
    """

    return 0.5*(pi*pi).sum() + action(phi)



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
    for _ in range(Nmd-1):
        pi += stepSize * force(phi).real
        phi+= stepSize * pi
    
    # final half step
    # p_{Nmd-1} = p_{Nmd-2} + \Delta t Re(dS/dphi)
    # x_{Nmd-1} = x_{Nmd-3/2} + 0.5 \Delta t * p_{Nmd-1}
    pi += stepSize * force(phi).real
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
            mean = torch.zeros((Nt,Nx),**torchTensorArgs).real, 
            std  = torch.ones((Nt,Nx),**torchTensorArgs).real
        ) + 0.j
        
        # Compute initial hamiltonian value
        S0 = action(phiM_n)
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
        phiM, logDetJ_NN = transformation(phiR)
        
        # Compute final hamiltonian value
        S1 = action(phiM)
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
    markovChainM = torch.zeros( (Nconf,Nt,Nx), **torchTensorArgs )
    # real plane configs
    markovChainR = torch.zeros( (Nconf,Nt,Nx), **torchTensorArgs )
    
    # create a list of action values
    action_full = torch.zeros( (Nconf+burnIn), **torchTensorArgs )
    action_markovChain = torch.zeros( (Nconf), **torchTensorArgs )
    
    # create a list of weights i.e. logDet J_NN
    logDetJ_NN_full = torch.zeros( (Nconf+burnIn), **torchTensorArgs )
    logDetJ_NN_markovChain = torch.zeros( (Nconf), **torchTensorArgs )
    
    # perform burnIn calculations
    phiR_n = phiR0.clone()
    phiM_n, logDetJ_NN_n = transformation(phiR_n)
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
        markovChainM[n+1],markovChainR[n+1],logDetJ_NN_markovChain[n+1],action_markovChain[n],acceptence =             hmc_step(phiM_n,phiR_n,logDetJ_NN_n)
        
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
        "S full": action_full,
        "log Det J_NN": logDetJ_NN_markovChain,
        "log Det J_NN full": logDetJ_NN_full,
        "acceptence rate": acceptenceRate/Nconf
    }



class LinearTransformation(torch.nn.Module):
    def __init__(self, Nt, Nx):
        super(LinearTransformation, self).__init__()
        self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros((Nt, Nx//2), **torchTensorArgs)))
        self.register_parameter(name='weight', param=torch.nn.Parameter(torch.zeros((Nt, Nx//2, Nt, Nx//2), **torchTensorArgs)))

    def forward(self, x):
        return torch.tensordot(x,self.weight,dims=([-1,-2],[0,1])) + self.bias
        



def LossMax(phi, logDet):
    n,Nt,Nx = phi.shape
    S = torch.zeros(n,**torchTensorArgs)
    for i in range(n):
        S[i] = action(phi[i,:,:])
    S -= logDet
    return torch.exp(-S).abs().mean()


def LossMin(phi, logDet):
    n,Nt,Nx = phi.shape
    S = torch.zeros(n,**torchTensorArgs)
    for i in range(n):
        S[i] = action(phi[i,:,:])
        
    return (S.imag - logDet.imag).abs().mean()



class complexRelu(torch.nn.Module):
    def __init__(self):
        super(complexRelu, self).__init__()
    def forward(self,z):
        return 0.5*(1+torch.cos(torch.angle(z)))*z



x=complexRelu()
z = torch.tensor(2+1j)
x(z)



lossfunctions = [LossMax,LossMin]
NLF = len(lossfunctions)



activationFunctions = [torch.nn.Softsign(), torch.nn.Tanh(),complexRelu()]
NAF = len(activationFunctions)



#%%time

import torch.optim as optim

EPOCHS = 100
epochs = np.arange(1,EPOCHS+1)

# GRID SEARCH

learningRates = np.linspace(0.001,0.002,25)
learnConfigurationNumbers = np.arange(1000,10000,1000)

NLR = len(learningRates)
NLCN = len(learnConfigurationNumbers)

Nconf_valid = 500
phivalid = torch.rand((Nconf_valid,Nt,Nx),**torchTensorArgs).real+1.j*tpOffset 

PRACL = [1,2,4]
internal = [2,4,8,16]
internalLayer=2

NPR = len(PRACL)
NINT = len(internal)

LOSSES = torch.zeros(NLF,NAF,NLCN, NLR, NPR,EPOCHS)
LossValid = torch.zeros(NLF,NAF,NLCN, NLR,NPR,EPOCHS)
BigLoss = torch.zeros(NLF,NAF,NLCN, NLR, NPR)

#for p in range(1): 
for p in range(NLF):   # loss function
    
    Loss = lossfunctions[p]
    if Loss == LossMax:
        value = True
        handle='LossMax'
    else:
        value = False
        handle='LossMin'

    #for n in range(1):    
    for n in range(1,NAF):   # nonlinear activation function

        nonlinearity = activationFunctions[n]
        def generate_parameters(internalLayer):

            layer = []
            for _ in range(internalLayer-1):

                d = LinearTransformation(Nt, Nx)
                layer.append(d)
                layer.append(nonlinearity)

            e = LinearTransformation(Nt, Nx)

            layer.append(e)

            return torch.nn.Sequential(*layer)

        def generate_coupling(internalLayer):    
            return AffineCoupling( m = generate_parameters(internalLayer), a = generate_parameters(internalLayer) )

        # ... training loop ...

        #for i in range(1):
        for i in range(NLCN):        # number of training configurations 

            Nconf_learn = learnConfigurationNumbers[i]
            phiR = torch.rand((Nconf_learn,Nt,Nx),**torchTensorArgs).real+1.j*tpOffset 
            for j in range(1):
            #for j in range(NLR):   # learning rate 

                LR = learningRates[j]
                #for k in range(1):
                for k in range(NPR):   # number of PRACL layers

                    numPRACLLayers = PRACL[k]

                    #for m in range(NINT):   # number of internal layers

                        #internalLayer = internal[m]

                    NN = []     # initialise the NN

                    for _ in range(numPRACLLayers):
                        NN.append(
                            PRCL(
                                Nt,
                                Nx,
                                coupling1 = generate_coupling(internalLayer),
                                coupling2 = generate_coupling(internalLayer),
                            )
                        )

                    NN = Sequential(NN)

                    optimizer = optim.Adam(NN.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, maximize=value)

                    for epoch in range(EPOCHS):

                        phiM, logDetJ_NN = NN(phiR)
                        loss = Loss(phiM, logDetJ_NN)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        LOSSES[p][n][i][j][k][epoch] = loss

                        if ((epoch+1) % 20 == 0) and (epoch != 0):

                            PATH = f"NN, Nconf = {Nconf_learn}, learning rate = {LR}, numPRACLLayers = {numPRACLLayers},internalLayer={internalLayer},nonlinearity={nonlinearity},lossfn={handle},{epoch} epochs.pdf"
                            torch.save(NN.state_dict(), PATH)
                        
                        if loss.item() >= 1e10:
                            ok = 0
                            LOSSES[p][n][i][j][k][(epoch+1):]=loss
                            break

                        #validation
                        with torch.no_grad():
                            PHIvalid, logD = NN(phivalid)
                            LossValid[p][n][i][j][k][epoch] = Loss(PHIvalid, logD)

                    BigLoss[p][n][i][j][k] = loss.item()

                    #plotting

                    fig = plt.figure(figsize=(7,5))
                    plt.plot(epochs, LOSSES[p][n][i][j][k].detach().numpy(),  '-r', label='Test loss function')
                    plt.yscale('log')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss function')

                    plt.plot(epochs, LossValid[p][n][i][j][k].detach().numpy(), '-b', label='Validation loss function')
                    plt.legend()

                    plt.title(f"Number of training configurations: {Nconf_learn} \n Learning rate: {LR}\n Number of PRACL layers: {numPRACLLayers}\n Number of internal layers: {internalLayer}\n Nonlinearity: {nonlinearity}\n lossfn: {handle}")
                    plt.savefig(f"Loss fn plot, Nconf_{Nconf_learn},lr_{LR}, NPRACL_{numPRACLLayers},Ninternal_{internalLayer}, Nonlinearity_{nonlinearity}, lossfn_{handle}.pdf",bbox_inches='tight', dpi=150)
                    plt.close('all')







