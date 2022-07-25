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

from lib_trainAnalysis import plot_loss, plot_actionStatistics, plot_fieldStatistics, plot_loss_eq,  plot_loss_eq2

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

        

if __name__ == "__main__":
    
    # Load the Hubbard Model
    HM = Hubbard2SiteModel(
        Nt = 16,
        beta = 4,
        U = 4,
        mu = 3,
        tangentPlaneOffset = -4.99933002e-01
    )
    
    # Grid search
    
    # Declare the list in which the hyperparameters range 

    list_numPRCLLayers = [1]
    list_numInternalLayers = [2,4,8]
    #list_activationfunctions = [torch.nn.Softsign, torch.nn.Tanh,complexRelu]
    list_activationfunctions = [torch.nn.Softsign]
    list_initEpsilons = [1e-12]
    list_trainEpsilons = [5e-3]
    list_trainBatchSizes = [4000]
    #list_trainMiniBatchSizes = [ 500,  550,  600,  650,  700,  750,  800,  850,  900,  950, 1000]
    list_trainMiniBatchSizes = [ 4000]
    list_lossFunctions = [MinimizeImaginaryPartLoss]
    list_learningRates = [5e-4]

    ok=0
    
    #.uniform_(-1e-5,1e-5)
    
    # Do the grid search

    for numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate in it.product(
            
            list_numPRCLLayers, list_numInternalLayers, list_activationfunctions, 
            list_initEpsilons, list_trainEpsilons, list_trainBatchSizes, 
            list_trainMiniBatchSizes, list_lossFunctions, list_learningRates
            ):
        
        # initialise the validation data
        validBatchSize = 500
        validData = torch.zeros((validBatchSize,HM.Nt,HM.Nx),**torchTensorArgs).uniform_(-trainEpsilon,trainEpsilon).real-4.99933002e-01j
       
        EPOCHS = 1500
        
        lossTrain = torch.zeros(2,EPOCHS)
        lossValid = torch.zeros(2,EPOCHS)
        lossTrainError = torch.zeros(2,EPOCHS)
        lossValidError = torch.zeros(2,EPOCHS)
        
        # generate the training data: a number of trainBatchSize configurations divided into minibatches,
            # with each minibatch having a number of trainMiniBatchSize configurations

        trainData = torch.utils.data.DataLoader(
            dataset = torch.utils.data.TensorDataset(
                torch.zeros((trainBatchSize,HM.Nt,HM.Nx), **torchTensorArgs).uniform_(-trainEpsilon,trainEpsilon).real -4.99933002e-01j
            ), 
            batch_size = trainMiniBatchSize, 
            shuffle    = True
        )
    
        for ok in range(2):
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


            if ok == 0:        
                NN = Equivariance(NN)
                okhandle = 'equivariance'
            else:
                okhandle = 'no equivariance'   
            #NN = x
            # pass the necessary parameters associated with the current loss function

            if lossFct == StatisticalPowerLoss:
                loss = lossFct(HM,147)
                optimizer = torch.optim.Adam(
                    NN.parameters(),
                    lr = learningRate,
                    maximize = True
                )
            else: 
                loss = lossFct(HM)
                handle = "MinimizeImaginaryPartLoss"
                optimizer = torch.optim.Adam(
                    NN.parameters(),
                    lr = learningRate,
                    maximize = False
                )

            # store all those hyperparameters in one dictionary for post-processing
            hyperparameters = {
              "numPRCLLayers": numPRCLLayers,
              "numInternalLayers": numInternalLayers,
              "activation": activation,
              "initEpsilon": initEpsilon,
              "trainEpsilon": trainEpsilon,
              "trainBatchSize": trainBatchSize,
              "trainMiniBatchSize": trainMiniBatchSize,
              "lossFct": handle,
              "learningRate": learningRate,
              #"okhandle": okhandle 
            }

            

            # train the NN and store the loss function evaluated on the training data 
            # and on the validation data after each epoch for post-processing




            NN, lossTrain[ok,...], lossValid[ok,...], lossTrainError[ok,...], lossValidError[ok,...], NNvalidData, logDetValid = train(epochs = EPOCHS, 
                NN = NN, 
                loss = loss, 
                optimizer = optimizer, 
                trainData = trainData, 
                validData = validData,
                hyperparameters = hyperparameters                                                                                                                       
            )

        
        
        # PLOTS
        
        # plot the two lossfunctions
        
        plot_loss_eq2(EPOCHS,lossTrain[0], lossTrain[1], lossValid[0], lossValid[1],lossTrainError[0], lossTrainError[1], lossValidError[0], lossValidError[1], hyperparameters)
        
        # loss function
        #plot_loss(lossTrain = lossTrain, lossValid = lossValid, hyperparameters=hyperparameters)

        # action statistics
        #plot_actionStatistics(validData = validData, NNvalidData = NNvalidData, logDetValid = logDetValid, hyperparameters = hyperparameters, HM=HM)
        
        # field statistics
        #plot_fieldStatistics(validData = validData, NNvalidData = NNvalidData, logDetValid = logDetValid, hyperparameters = hyperparameters, HM=HM)





