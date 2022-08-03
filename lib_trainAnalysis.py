import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


from lib_2SiteModel import Hubbard2SiteModelIsleIsleAction


def averagePhi(phi):
    r"""
        This computes the average of the configuration over the space time 
        volumen 

        \f[
            \frac{1}{V} \sum_{t,x} \phi_{t,x}
        \f] 
    """
    _, Nt, Nx = phi.shape
    
    return phi.sum(dim=(-1,-2))/(Nt*Nx)

def modPhi(phi): 
    r"""
        This computes the L2 norm of the configuration over the space time
        volume

        \f[
            |\phi| = \sqrt{ \sum_{t,x} \phi_{t,x}^2 }
        \f]
    """
    _, Nt, Nx = phi.shape

    return torch.sqrt((torch.square(phi)).sum(dim=(-1,-2)))
        
def decodeHyperparams(hyperparameters):
    
    numPRCLLayers =     hyperparameters["numPRCLLayers"]
    numInternalLayers = hyperparameters["numInternalLayers"]
    activation = hyperparameters["activation"]()
    initEpsilon = hyperparameters["initEpsilon"]
    trainEpsilon = hyperparameters["trainEpsilon"]
    trainBatchSize = hyperparameters["trainBatchSize"]
    trainMiniBatchSize = hyperparameters["trainMiniBatchSize"]
    lossFct = hyperparameters["lossFct"]
    learningRate = hyperparameters["learningRate"]
    Nt = hyperparameters["Nt"]
    #okhandle = hyperparameters["okhandle"]
    if 'epoch' in hyperparameters:
        epoch = hyperparameters['epoch']
        return Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch
    else:
        return Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate
    
    
def plotTitle(hyperparameters):
    if 'epoch' in hyperparameters:
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch = decodeHyperparams(hyperparameters)
        return  f"Nt: {Nt}\ninitEpsilon: {initEpsilon}\ntrainEpsilon: {trainEpsilon}\ntrainBatchSize: {trainBatchSize}\ntrainMiniBatchSize: {trainMiniBatchSize}\nlearningRate: {learningRate}\nnumPRCLLayers: {numPRCLLayers}\nnumInternalLayers: {numInternalLayers}\nactivation: {activation}\nlossfn: {lossFct}\nepochs: {epoch}"
    else:
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate = decodeHyperparams(hyperparameters)
        return  f"Nt: {Nt}\ninitEpsilon: {initEpsilon}\ntrainEpsilon: {trainEpsilon}\ntrainBatchSize: {trainBatchSize}\ntrainMiniBatchSize: {trainMiniBatchSize}\nlearningRate: {learningRate}\nnumPRCLLayers: {numPRCLLayers}\nnumInternalLayers: {numInternalLayers}\nactivation: {activation}\nlossfn: {lossFct}\n"


def pathname(hyperparameters):
    
    if 'epoch' in hyperparameters:
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch = decodeHyperparams(hyperparameters)
        return f",Nt_{Nt}initEpsilon_{initEpsilon},trainEpsilon_{trainEpsilon},trainBatchSize_{trainBatchSize},trainMiniBatchSize_{trainMiniBatchSize},learningRate_{learningRate}, numPRCLLayers_{numPRCLLayers},numInternalLayers_{numInternalLayers}, activation_{activation}, lossfn_{lossFct}, epochs_{epoch}.pdf"
    else:
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate = decodeHyperparams(hyperparameters)
        return f",Nt_{Nt}initEpsilon_{initEpsilon},trainEpsilon_{trainEpsilon},trainBatchSize_{trainBatchSize},trainMiniBatchSize_{trainMiniBatchSize},learningRate_{learningRate}, numPRCLLayers_{numPRCLLayers},numInternalLayers_{numInternalLayers}, activation_{activation}, lossfn_{lossFct}.pdf"
    
    
def pathnameNN(hyperparameters):
    
    if 'epoch' in hyperparameters:
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch = decodeHyperparams(hyperparameters)
        return f",Nt_{Nt}initEpsilon_{initEpsilon},trainEpsilon_{trainEpsilon},trainBatchSize_{trainBatchSize},trainMiniBatchSize_{trainMiniBatchSize},learningRate_{learningRate}, numPRCLLayers_{numPRCLLayers},numInternalLayers_{numInternalLayers}, activation_{activation}, lossfn_{lossFct}, epochs_{epoch}.pt"
    else:
        Nt, numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate = decodeHyperparams(hyperparameters)
        return f",Nt_{Nt}initEpsilon_{initEpsilon},trainEpsilon_{trainEpsilon},trainBatchSize_{trainBatchSize},trainMiniBatchSize_{trainMiniBatchSize},learningRate_{learningRate}, numPRCLLayers_{numPRCLLayers},numInternalLayers_{numInternalLayers}, activation_{activation}, lossfn_{lossFct}.pt"
    
def plot_loss_eq2(EPOCHS,lossTrainEq, lossTrainNoEq, lossValidEq, lossValidNoEq,lossTrainEqErr, lossTrainNoEqErr, lossValidEqErr, lossValidNoEqErr, hyperparameters):
    
    W = 0.5
    H = 0.4
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5), sharey = True)
    #outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    
    # train data
    
    #inner = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0], wspace=W, hspace=H)
    #ax0 = plt.Subplot(fig, inner[0])
    #fig.add_subplot(ax0)
    
    epochs = np.arange(0,EPOCHS)
    
    ax0.errorbar(epochs,lossTrainEq, yerr = lossTrainEqErr,color='r', capsize=2, label = "Equivariance", fmt='x')
    ax0.errorbar(epochs,lossTrainNoEq, yerr = lossTrainNoEqErr,color='b', capsize=2,  label = "No equivariance", fmt='x')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss function')
    ax0.set_title('Train data')
    ax0.set_yscale('log')
    ax0.legend(loc="best")

    # valid data
    
    #ax1 = plt.Subplot(fig, inner[1])
    #fig.add_subplot(ax1)

    ax1.errorbar(epochs,lossValidEq,yerr=lossValidEqErr, color='r',   capsize=2, label = "Equivariance", fmt='x')
    ax1.errorbar(epochs,lossValidNoEq, yerr=lossValidNoEqErr,color='b',   capsize=2, label = "No equivariance", fmt='x')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss function')
    ax1.set_title('Validation data')
    ax1.set_yscale('log')
    ax1.legend(loc="best")
    
    fig.tight_layout()
    fig.suptitle(f"SPACETIME SYMMETRY  Loss fn plots\n{plotTitle(hyperparameters)}",y=1.4)
    fig.savefig(f"SPACETIME SYMMETRY Loss fn plots{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 
    
    
def plot_loss_eq3(EPOCHS,lossTrainEq, lossTrainNoEq, lossValidEq, lossValidNoEq,lossTrainEqErr, lossTrainNoEqErr, lossValidEqErr, lossValidNoEqErr, hyperparameters):
    
    W = 0.5
    H = 0.4
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5), sharey = True)
    #outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    
    # train data
    
    #inner = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0], wspace=W, hspace=H)
    #ax0 = plt.Subplot(fig, inner[0])
    #fig.add_subplot(ax0)
    
    epochs = np.arange(0,EPOCHS)
    
    ax0.errorbar(epochs,lossTrainEq, yerr = lossTrainEqErr,color='r', capsize=2, label = "Equivariance")
    ax0.errorbar(epochs,lossTrainNoEq, yerr = lossTrainNoEqErr,color='b', capsize=2,  label = "No equivariance")
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss function')
    ax0.set_title('Train data')
    ax0.set_yscale('log')
    ax0.legend(loc="best")

    # valid data
    
    #ax1 = plt.Subplot(fig, inner[1])
    #fig.add_subplot(ax1)

    ax1.errorbar(epochs,lossValidEq,yerr=lossValidEqErr, color='r',   capsize=2, label = "Equivariance")
    ax1.errorbar(epochs,lossValidNoEq, yerr=lossValidNoEqErr,color='b',   capsize=2, label = "No equivariance")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss function')
    ax1.set_title('Validation data')
    ax1.set_yscale('log')
    ax1.legend(loc="best")
    
    fig.tight_layout()
    fig.suptitle(f"SPACETIME SYMMETRY  Loss fn plots\n{plotTitle(hyperparameters)}",y=1.4)
    fig.savefig(f"Results/SPACETIME SYMMETRY Loss fn plots{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 
    
def plot_loss_eq(lossTrainEq, lossTrainNoEq, lossValidEq, lossValidNoEq, hyperparameters):
    
    W = 0.5
    H = 0.4
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
    #outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    
    # train data
    
    
    ax0.plot(lossTrainEq, 'r', label = "Equivariance")
    ax0.plot(lossTrainNoEq, 'b', label = "No equivariance")
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss function')
    ax0.set_title('Train data')
    ax0.set_yscale('log')
    ax0.legend(loc="lower center")

    # valid data
    

    ax1.plot(lossValidEq, 'r', label = "Equivariance")
    ax1.plot(lossValidNoEq, 'b', label = "No equivariance")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss function')
    ax1.set_title('Validation data')
    ax1.set_yscale('log')
    ax1.legend(loc="lower center")
    
    fig.tight_layout()
    fig.suptitle(f"Loss fn plots{plotTitle(hyperparameters)}",y=1.25)
    fig.savefig(f"Loss fn plots{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 


def plot_loss(lossTrain, lossValid, lossTrainError,lossValidError, hyperparameters):
    
    EPOCHS = len(lossTrain)
    epochs = np.arange(0,EPOCHS)
    
    plt.errorbar(epochs,lossTrain, yerr = lossTrainError,color='r', capsize=2, label = "Train Loss")
    plt.errorbar(epochs,lossValid, yerr = lossValidError,color='b', capsize=2, label = "Valid Loss")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss function')
    plt.title(f"Loss function plot equivariance\n{plotTitle(hyperparameters)}", y=1.01)
    plt.yscale('log')
    plt.legend(loc = 'best')
    savepath = f"Loss fn plot equivariance{pathname(hyperparameters)}"
    plt.savefig(savepath, bbox_inches='tight', dpi=150)
    plt.clf()
    plt.close()
    

class ACTION(torch.nn.Module):
    def __init__(self, Hubbard2SiteModelIsleIsleAction):
        super(ACTION,self).__init__()

        self.HM = Hubbard2SiteModelIsleIsleAction

    def forward(self,phi):

        return self.HM.calculate_batch_action(phi)
    
    
    
def get_x_y_values(validData, NNvalidData, logDetValid, HM):

    x = ACTION(HM)
    S_validData = x(validData)
    S_NNvalidData =x(NNvalidData)


    diffSvalid = S_NNvalidData - S_validData
    
    y1phi = S_validData.real.detach().numpy()
    y1NNphi =  S_NNvalidData.real.detach().numpy()

    x1phi = (averagePhi(validData)).real.detach().numpy()
    x1NNphi = (averagePhi(NNvalidData)).real.detach().numpy()

    x2phi = (averagePhi(validData)).imag.detach().numpy()
    x2NNphi = (averagePhi(NNvalidData)).imag.detach().numpy()

    x3phi = (modPhi(validData.real)).detach().numpy()
    x3NNphi = (modPhi(NNvalidData.real)).detach().numpy()

    x4phi = (modPhi(validData.imag)).detach().numpy()
    x4NNphi = (modPhi(NNvalidData.imag)).detach().numpy()
    
    y3phi = y1NNphi - logDetValid.real.detach().numpy()
    
    y2phi = S_validData.imag.detach().numpy()%(2*np.pi)
    y2NNphi = S_NNvalidData.imag.detach().numpy()%(2*np.pi)
    
    y4phi = (S_NNvalidData.imag.detach().numpy()- logDetValid.imag.detach().numpy())%(2*np.pi)
    
    return S_NNvalidData, S_validData, diffSvalid, y1phi, y1NNphi, x1phi, x1NNphi, x2phi, x2NNphi, x3phi, x3NNphi, x4phi, x4NNphi, y3phi, y2phi, y2NNphi, y4phi
    
    
def plot_actionStatistics(validData, NNvalidData, logDetValid, hyperparameters, HM, okhandle):
    
    # get the values for the x and y axes
    
    S_NNvalidData, S_validData, diffSvalid, y1phi, y1NNphi, x1phi, x1NNphi, x2phi, x2NNphi, x3phi,\
                        x3NNphi, x4phi, x4NNphi, y3phi, y2phi, y2NNphi, y4phi = get_x_y_values(validData, NNvalidData, logDetValid, HM)
    
    # create the large figure where all the actions plots will be displayed
    
    W = 0.5
    H = 0.4

    fig = plt.figure(figsize=(12, 28))
    outer = gridspec.GridSpec(7, 1, wspace=0.2, hspace=0.2,height_ratios=[1,1,1,2,2,2,2])

    # first subplot: i=0. Safety checks
    
    # Add the two sub-subplots on the left and right
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[0], wspace=W, hspace=H)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(ax1)
    
    # real part of S should increase - or not

    ax0.plot(diffSvalid.real.detach().numpy())
    ax0.set_xlabel('n')
    ax0.set_ylabel('Re(S(NNphi)-S(phi))')
    ax0.set_title('Re(S(NNphi)-S(phi))')

    # imag part of S should be const

    ax1.plot(diffSvalid.imag.detach().numpy())
    ax1.set_xlabel('n')
    ax1.set_ylabel('Im(SNN(phi))-Im(S(phi))')
    ax1.set_title('Im(SNN(phi))-Im(S(phi))')
    
    # second subplot: log det J
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[1], wspace=W, hspace=H)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(ax1)
    
    # real part of S should increase - or not

    ax0.plot(logDetValidEq.real.detach().numpy())
    ax0.set_xlabel('n')
    ax0.set_ylabel('Re(log det J)')
    ax0.set_title('Re(log det J)')
    

    # imag part of S should be const

    ax1.plot(logDetValidEq.imag.detach().numpy())
    ax1.set_xlabel('n')
    ax1.set_ylabel('Im(log det J)')
    ax1.set_title('Im(log det J)')
    
    

    # second subplot: i=1. y = Re S
    
    # add the four sub-subplots 

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[1], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)
    
    # plot

    axs0.plot(x1phi, y1phi, '.r', label = 'phi')
    axs0.plot(x1NNphi, y1NNphi, '.b', label = 'NN(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Re S')
    axs0.set_title('Re S for phi and NN(phi)')


    axs1.plot(x2phi,y1phi, '.r', label = 'phi')
    axs1.plot(x2NNphi, y1NNphi, '.b', label = 'NN(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Re S')
    axs1.set_title('Re S for phi and NN(phi)')


    axs2.plot(x3phi, y1phi, '.r', label = 'phi')
    axs2.plot(x3NNphi, y1NNphi, '.b', label = 'NN(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Re S')
    axs2.set_title('Re S for phi and NN(phi)')

    axs3.plot(x4phi, y1phi, '.r', label = 'phi')
    axs3.plot(x4NNphi, y1NNphi, '.b', label = 'NN(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Re S')
    axs3.set_title('Re S for phi and NN(phi)')


    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)

    
     # third subplot: i=2. y = Re Seff

     # add the four sub-subplots   
        
    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[2], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)
    
    
    # plot

    axs0.plot(x1phi, y3phi, '.r', label = 'Seff(phi)')
    axs0.plot(x1phi, y1phi, '.b', label = 'S(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Re S or Re Seff')
    axs0.set_title('Re S(phi) vs Seef(phi)')

    axs1.plot(x2phi,y3phi, '.r', label = 'Seff(phi)')
    axs1.plot(x2phi,y1phi, '.b', label = 'S(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Re S or Re Seff')
    axs1.set_title('Re S(phi) vs Seef(phi)')

    axs2.plot(x3phi, y3phi, '.r', label = 'Seff(phi)')
    axs2.plot(x3phi, y1phi, '.b', label = 'S(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Re S or Re Seff')
    axs2.set_title('Re S(phi) vs Seef(phi)')

    axs3.plot(x4phi, y3phi, '.r', label = 'Seff(phi)')
    axs3.plot(x4phi, y1phi, '.b', label = 'S(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Re S or Re Seff')
    axs3.set_title('Re S(phi) vs Seef(phi)') 

    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)

    # Fourth subplot: i=3. Im S mod 2pi
    
    # create the 4 subplots

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[3], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)
    
    # plot
    
    axs0.plot(x1phi, y2phi, '.r', label = 'phi')
    axs0.plot(x1NNphi, y2NNphi, '.b', label = 'NN(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Im S mod 2π')
    axs0.set_title('Im S mod 2π for phi and NN(phi)')

    axs1.plot(x2phi,y2phi, '.r', label = 'phi')
    axs1.plot(x2NNphi, y2NNphi, '.b', label = 'NN(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Im S mod 2π')
    axs1.set_title('Im S mod 2π for phi and NN(phi)')

    axs2.plot(x3phi, y2phi, '.r', label = 'phi')
    axs2.plot(x3NNphi, y2NNphi, '.b', label = 'NN(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Im S mod 2π')
    axs2.set_title('Im S mod 2π for phi and NN(phi)')

    axs3.plot(x4phi, y2phi, '.r', label = 'phi')
    axs3.plot(x4NNphi, y2NNphi, '.b', label = 'NN(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Im S mod 2π')
    axs3.set_title('Im S mod 2π for phi and NN(phi)')


    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)
    
    
    # 5th plot: i=4. y = Im Seff mod 2 pi or Im S mod 2pi
    # Seff
    
    # create the 4 subplots

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[4], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)


    # plot
    
    axs0.plot(x1phi, y4phi, '.r', label = 'Seff(phi)')
    axs0.plot(x1phi, y2phi, '.b', label = 'S(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Im Seff or S vs mod 2π')
    axs0.set_title('Im S mod 2π vs Im Seff mod 2π')

    axs1.plot(x2phi,y4phi, '.r', label = 'Seff(phi)')
    axs1.plot(x2phi,y2phi, '.b', label = 'S(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Im Seff or S mod 2π')
    axs1.set_title('Im S mod 2π vs Im Seff mod 2π')

    axs2.plot(x3phi, y4phi, '.r', label = 'Seff(phi)')
    axs2.plot(x3phi, y2phi, '.b', label = 'S(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Im Seff or S mod 2π')
    axs2.set_title('Im S mod 2π vs Im Seff mod 2π')

    axs3.plot(x4phi, y4phi, '.r', label = 'Seff(phi)')
    axs3.plot(x4phi, y2phi, '.b', label = 'S(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Im Seff or S mod 2π')
    axs3.set_title('Im S mod 2π vs Im Seff mod 2π')


    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)

    fig.suptitle(f"S statistics{plotTitle(hyperparameters)}",y=0.95)
    fig.savefig(f"S statistics{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 
    
def plot_actionStatistics2(validDataEq, NNvalidDataEq, logDetValidEq, hyperparameters, HM):
    
    # get the values for the x and y axes
    
     # with equivariance
    
    S_NNvalidDataEq, S_validDataEq, diffSvalidEq, y1phiEq, y1NNphiEq, x1phiEq, x1NNphiEq, x2phiEq, x2NNphiEq, x3phiEq,\
                        x3NNphiEq, x4phiEq, x4NNphiEq, y3phiEq, y2phiEq, y2NNphiEq, y4phiEq = get_x_y_values(validDataEq, NNvalidDataEq, logDetValidEq, HM)
    
    
    
    # create the large figure where all the actions plots will be displayed
    
    W = 0.4
    H = 0.6

    fig = plt.figure(figsize=(25, 20))
    outer = gridspec.GridSpec(7, 1, wspace=0.2, hspace=0.2,height_ratios=[1,1,1,2,2,2,2])
    
        
    # first subplot: i=0. Safety checks - ACTION
    # equivariance
    
    # Add the two sub-subplots on the left and right
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[0], wspace=W, hspace=H)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(ax1)
    
    
    # real part of S should increase - or not

    ax0.plot(S_NNvalidDataEq.real.detach().numpy())
    ax0.set_xlabel('n')
    ax0.set_ylabel('Re(S(NNphi))')
    ax0.set_title('Re(S(NNphi))')
    

    # imag part of S should be const

    ax1.plot(S_NNvalidDataEq.imag.detach().numpy())
    ax1.set_xlabel('n')
    ax1.set_ylabel('Im(SNN(phi))')
    ax1.set_title('Im(SNN(phi))')
    
   
    
    # subplot: i=1. logDetJ
    # equivariance
    
    # Add the two sub-subplots on the left and right
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[1], wspace=W, hspace=H)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(ax1)
    
    # real part of S should increase - or not

    ax0.plot(logDetValidEq.real.detach().numpy())
    ax0.set_xlabel('n')
    ax0.set_ylabel('Re(log det J)')
    ax0.set_title('Re(log det J)')
    

    # imag part of S should be const

    ax1.plot(logDetValidEq.imag.detach().numpy())
    ax1.set_xlabel('n')
    ax1.set_ylabel('Im(log det J)')
    ax1.set_title('Im(log det J)')
    
    
    
    # subplot: i=4. Safety checks - EFFECTIVE ACTION
    # equivariance
    
    # Add the two sub-subplots on the left and right
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[2], wspace=W, hspace=H)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    ax1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(ax1)
    
    SeffEq = S_NNvalidDataEq-logDetValidEq
    SeffNoEq = S_NNvalidDataNoEq-logDetValidNoEq
    
    # real part of S should increase - or not

    ax0.plot(SeffEq.real.detach().numpy())
    ax0.set_xlabel('n')
    ax0.set_ylabel('Re(Seff(phi))')
    ax0.set_title('Re(Seff(phi))')
    

    # imag part of S should be const

    ax1.plot(SeffEq.imag.detach().numpy())
    ax1.set_xlabel('n')
    ax1.set_ylabel('Im(Seff(phi))')
    ax1.set_title('Im(Seff(phi))')
    
    
    
    

    # subplot: i=4. y = Re S
    # equivariance
    
    # add the four sub-subplots 

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[3], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)
    
    # plot

    axs0.plot(x1phiEq, y1phiEq, '.r', label = 'phi')
    axs0.plot(x1NNphiEq, y1NNphiEq, '.b', label = 'NN(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Re S')
    axs0.set_title('Re S for phi and NN(phi)')


    axs1.plot(x2phiEq,y1phiEq, '.r', label = 'phi')
    axs1.plot(x2NNphiEq, y1NNphiEq, '.b', label = 'NN(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Re S')
    axs1.set_title('Re S for phi and NN(phi)')


    axs2.plot(x3phiEq, y1phiEq, '.r', label = 'phi')
    axs2.plot(x3NNphiEq, y1NNphiEq, '.b', label = 'NN(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Re S')
    axs2.set_title('Re S for phi and NN(phi)')

    axs3.plot(x4phiEq, y1phiEq, '.r', label = 'phi')
    axs3.plot(x4NNphiEq, y1NNphiEq, '.b', label = 'NN(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Re S')
    axs3.set_title('Re S for phi and NN(phi)')


    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)

   

    
     # third subplot: i=6. y = Re Seff
     # equivariance 
     # add the four sub-subplots   
        
    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[4], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)
    
    
    # plot

    axs0.plot(x1phiEq, y3phiEq, '.r', label = 'Seff(phi)')
    axs0.plot(x1phiEq, y1phiEq, '.b', label = 'S(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Re S or Re Seff')
    axs0.set_title('Re S(phi) vs Seef(phi)')

    axs1.plot(x2phiEq,y3phiEq, '.r', label = 'Seff(phi)')
    axs1.plot(x2phiEq,y1phiEq, '.b', label = 'S(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Re S or Re Seff')
    axs1.set_title('Re S(phi) vs Seef(phi)')

    axs2.plot(x3phiEq, y3phiEq, '.r', label = 'Seff(phi)')
    axs2.plot(x3phiEq, y1phiEq, '.b', label = 'S(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Re S or Re Seff')
    axs2.set_title('Re S(phi) vs Seef(phi)')

    axs3.plot(x4phiEq, y3phiEq, '.r', label = 'Seff(phi)')
    axs3.plot(x4phiEq, y1phiEq, '.b', label = 'S(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Re S or Re Seff')
    axs3.set_title('Re S(phi) vs Seef(phi)') 

    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)
    
    
    # Fourth subplot: i=8. Im S mod 2pi
    
    # create the 4 subplots

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[5], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)
    
    # plot
    
    axs0.plot(x1phiEq, y2phiEq, '.r', label = 'phi')
    axs0.plot(x1NNphiEq, y2NNphiEq, '.b', label = 'NN(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Im S mod 2π')
    axs0.set_title('Im S mod 2π for phi and NN(phi)')

    axs1.plot(x2phiEq,y2phiEq, '.r', label = 'phi')
    axs1.plot(x2NNphiEq, y2NNphiEq, '.b', label = 'NN(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Im S mod 2π')
    axs1.set_title('Im S mod 2π for phi and NN(phi)')

    axs2.plot(x3phiEq, y2phiEq, '.r', label = 'phi')
    axs2.plot(x3NNphiEq, y2NNphiEq, '.b', label = 'NN(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Im S mod 2π')
    axs2.set_title('Im S mod 2π for phi and NN(phi)')

    axs3.plot(x4phiEq, y2phiEq, '.r', label = 'phi')
    axs3.plot(x4NNphiEq, y2NNphiEq, '.b', label = 'NN(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Im S mod 2π')
    axs3.set_title('Im S mod 2π for phi and NN(phi)')


    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)
    
    
    
    
    # 5th plot: i=10. y = Im Seff mod 2 pi or Im S mod 2pi
    # Seff
    
    # create the 4 subplots

    inner = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=outer[6], wspace=W, hspace=H)
    axs0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(axs0)

    axs1 = plt.Subplot(fig, inner[1])
    fig.add_subplot(axs1)

    axs2 = plt.Subplot(fig, inner[2])
    fig.add_subplot(axs2)

    axs3 = plt.Subplot(fig, inner[3])
    fig.add_subplot(axs3)


    # plot
    
    axs0.plot(x1phiEq, y4phiEq, '.r', label = 'Seff(phi)')
    axs0.plot(x1phiEq, y2phiEq, '.b', label = 'S(phi)')
    axs0.set_xlabel('Re <phi>')
    axs0.set_ylabel('Im Seff or S vs mod 2π')
    axs0.set_title('Im S mod 2π vs Im Seff mod 2π')

    axs1.plot(x2phiEq,y4phiEq, '.r', label = 'Seff(phi)')
    axs1.plot(x2phiEq,y2phiEq, '.b', label = 'S(phi)')
    axs1.set_xlabel('Im <phi>')
    axs1.set_ylabel('Im Seff or S mod 2π')
    axs1.set_title('Im S mod 2π vs Im Seff mod 2π')

    axs2.plot(x3phiEq, y4phiEq, '.r', label = 'Seff(phi)')
    axs2.plot(x3phiEq, y2phiEq, '.b', label = 'S(phi)')
    axs2.set_xlabel('|Re phi|')
    axs2.set_ylabel('Im Seff or S mod 2π')
    axs2.set_title('Im S mod 2π vs Im Seff mod 2π')

    axs3.plot(x4phiEq, y4phiEq, '.r', label = 'Seff(phi)')
    axs3.plot(x4phiEq, y2phiEq, '.b', label = 'S(phi)')
    axs3.set_xlabel('|Im phi|')
    axs3.set_ylabel('Im Seff or S mod 2π')
    axs3.set_title('Im S mod 2π vs Im Seff mod 2π')


    axs0.legend(ncol=1)
    axs1.legend(ncol=1)
    axs2.legend(ncol=1)
    axs3.legend(ncol=1)
    
    
    

    fig.suptitle(f"S statistics, Equivariance{plotTitle(hyperparameters)}",y=0.95)
   
    fig.savefig(f"S statistics, Equivariance{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 
    
    

def plot_fieldStatistics2(validDataEq, NNvalidDataEq, logDetValidEq, hyperparameters, HM):
    
    # get the values for the x and y axes
    
    # with equivariance
    
    S_NNvalidDataEq, S_validDataEq, diffSvalidEq, y1phiEq, y1NNphiEq, x1phiEq, x1NNphiEq, x2phiEq, x2NNphiEq, x3phiEq,\
                        x3NNphiEq, x4phiEq, x4NNphiEq, y3phiEq, y2phiEq, y2NNphiEq, y4phiEq = get_x_y_values(validDataEq, NNvalidDataEq, logDetValidEq, HM)
   
    
    # create the large figure where all the field plots will be displayed
    fig = plt.figure(figsize=(14, 12))
    outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.3)

    # i=0
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[0], wspace=0.1, hspace=0.2)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    # Re <phi> vs Im <phi> for equivariance
    yphi = x1phiEq
    yNNphi = x1NNphiEq

    xphi = x2phiEq
    xNNphi = x2NNphiEq

    ax0.plot(xphi,yphi, '.r', label = 'phi')
    ax0.plot(xNNphi,yNNphi, '.b', label = 'NN(phi)')
    ax0.set_xlabel('Im <phi>')
    ax0.set_ylabel('Re <phi>')
    ax0.legend()
    ax0.set_title('Equivariance')
    
    # i=1
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[1], wspace=0.1, hspace=0.2)
    ax1 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax1)

    # Re <phi> vs Im <phi> for no equivariance
    yphi = x1phiNoEq
    yNNphi = x1NNphiNoEq

    xphi = x2phiNoEq
    xNNphi = x2NNphiNoEq

    ax1.plot(xphi,yphi, '.r', label = 'phi')
    ax1.plot(xNNphi,yNNphi, '.b', label = 'NN(phi)')
    ax1.set_xlabel('Im <phi>')
    ax1.set_ylabel('Re <phi>')
    ax1.legend()
    ax1.set_title('No Equivariance')

    # i=2
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[2], wspace=0.1, hspace=0.1)
    ax2 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax2)

    # mod Re phi vs Im mod phi for equivariance

    yphi = x3phiEq
    yNNphi = x3NNphiEq

    xphi = x4phiEq
    xNNphi = x4NNphiEq
    ax2.scatter(xphi,yphi, c='r', marker='.',label = 'phi')
    ax2.scatter(xNNphi,yNNphi, c='b', marker='.',label = 'NN(phi)')
    ax2.set_xlabel('|Im phi|')
    ax2.set_ylabel('|Re phi|')
    ax2.legend()
    
     # i=3
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[3], wspace=0.1, hspace=0.1)
    ax3 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax3)

    # mod Re phi vs Im mod phi for no equivariance

    yphi = x3phiNoEq
    yNNphi = x3NNphiNoEq

    xphi = x4phiNoEq
    xNNphi = x4NNphiNoEq
    ax3.scatter(xphi,yphi, c='r', marker='.',label = 'phi')
    ax3.scatter(xNNphi,yNNphi, c='b', marker='.',label = 'NN(phi)')
    ax3.set_xlabel('|Im phi|')
    ax3.set_ylabel('|Re phi|')
    ax3.legend()


    # i=4
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[4], wspace=0.1, hspace=0.1)
    ax4 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax4)

    # Re phi vs Im phi for equivariance
    
    # get some new data
    yphi = validDataEq.sum(dim=(-1,-2)).real.detach().numpy()
    yNNphi = NNvalidDataEq.sum(dim=(-1,-2)).real.detach().numpy()

    xphi = validDataEq.sum(dim=(-1,-2)).imag.detach().numpy()
    xNNphi = NNvalidDataEq.sum(dim=(-1,-2)).imag.detach().numpy()
    
    #plot 
    
    ax4.scatter(xphi,yphi, c='r',marker='.',  label = 'phi')
    ax4.scatter(xNNphi,yNNphi, c='b', marker='.',label = 'NN(phi)')
    ax4.set_xlabel('Im phi')
    ax4.set_ylabel('Re phi')
    ax4.legend()
    
    
    # i=5
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[5], wspace=0.1, hspace=0.1)
    ax5 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax5)

    # Re phi vs Im phi for no equivariance
    
    # get some new data
    yphi = validDataNoEq.sum(dim=(-1,-2)).real.detach().numpy()
    yNNphi = NNvalidDataNoEq.sum(dim=(-1,-2)).real.detach().numpy()

    xphi = validDataNoEq.sum(dim=(-1,-2)).imag.detach().numpy()
    xNNphi = NNvalidDataNoEq.sum(dim=(-1,-2)).imag.detach().numpy()
    
    #plot 
    
    ax5.scatter(xphi,yphi, c='r',marker='.',  label = 'phi')
    ax5.scatter(xNNphi,yNNphi, c='b', marker='.',label = 'NN(phi)')
    ax5.set_xlabel('Im phi')
    ax5.set_ylabel('Re phi')
    ax5.legend()
    
    
    fig.suptitle(f"Field statistics,Equivariance\n{plotTitle(hyperparameters)}",y=1.05)

    fig.savefig(f"Field statistics,Equivariance{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 
    