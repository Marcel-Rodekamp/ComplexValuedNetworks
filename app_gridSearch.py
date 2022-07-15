import numpy as np
import torch

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import itertools as it


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

from lib_activations import complexRelu

from lib_2SiteModel import Hubbard2SiteModel

from lib_loss import StatisticalPowerLoss,MinimizeImaginaryPartLoss

from lib_train import train

from lib_trainAnalysis import plot_loss

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
            - HM:              Hubbard2SiteModel instance carrying information to set up the PRCL layers  
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

if __name__ == "__main__":
    HM = Hubbard2SiteModel(
        Nt = 4,
        beta = 4,
        U = 4,
        mu = 3,
        tangentPlaneOffset = -1.99973201
    )

    list_numPRCLLayers = [1]
    list_numInternalLayers = [2]
    list_activationfunctions = [complexRelu]
    list_initEpsilons = [1e-4]
    list_trainEpsilons = [1e-5]
    list_trainBatchSizes = [1000]
    list_trainMiniBatchSizes = [100]
    list_lossFunctions = [MinimizeImaginaryPartLoss, StatisticalPowerLoss]
    list_learningRates = [1e-4]

    validBatchSize = 100
    validData = torch.rand((validBatchSize,HM.Nt,HM.Nx),**torchTensorArgs)

    for numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate in it.product(
            list_numPRCLLayers, list_numInternalLayers, list_activationfunctions, 
            list_initEpsilons, list_trainEpsilons, list_trainBatchSizes, 
            list_trainMiniBatchSizes, list_lossFunctions, list_learningRates
            ):
        
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

        if lossFct == StatisticalPowerLoss:
            loss = lossFct(HM,147)
            optimizer = torch.optim.Adam(
                NN.parameters(),
                lr = learningRate,
                maximize = True
            )
        else: 
            loss = lossFct(HM)
            optimizer = torch.optim.Adam(
                NN.parameters(),
                lr = learningRate
            )
        
        trainData = torch.utils.data.DataLoader(
            dataset = torch.utils.data.TensorDataset(
                torch.zeros((trainBatchSize,HM.Nt,HM.Nx), **torchTensorArgs).uniform_(-trainEpsilon,trainEpsilon)
            ), 
            batch_size = trainMiniBatchSize, 
            shuffle    = True
        )

        NN, lossTrain, lossValid = train(epochs = 100, 
            NN = NN, 
            loss = loss, 
            optimizer = optimizer, 
            trainData = trainData, 
            validData = validData
        )

        plot_loss(lossTrain = lossTrain, lossValid = lossValid)


