import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


from lib_2SiteModel import Hubbard2SiteModel


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
    #okhandle = hyperparameters["okhandle"]
    if 'epoch' in hyperparameters:
        epoch = hyperparameters['epoch']
        return numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch
    else:
        return numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate
    
    
def plotTitle(hyperparameters):
    if 'epoch' in hyperparameters:
        numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch = decodeHyperparams(hyperparameters)
        return  f"initEpsilon: {initEpsilon}\ntrainEpsilon: {trainEpsilon}\ntrainBatchSize: {trainBatchSize}\ntrainMiniBatchSize: {trainMiniBatchSize}\nlearningRate: {learningRate}\nnumPRCLLayers: {numPRCLLayers}\nnumInternalLayers: {numInternalLayers}\nactivation: {activation}\nlossfn: {lossFct}\nepochs: {epoch}"
    else:
        numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate = decodeHyperparams(hyperparameters)
        return  f"initEpsilon: {initEpsilon}\ntrainEpsilon: {trainEpsilon}\ntrainBatchSize: {trainBatchSize}\ntrainMiniBatchSize: {trainMiniBatchSize}\nlearningRate: {learningRate}\nnumPRCLLayers: {numPRCLLayers}\nnumInternalLayers: {numInternalLayers}\nactivation: {activation}\nlossfn: {lossFct}\n"


def pathname(hyperparameters):
    
    if 'epoch' in hyperparameters:
        numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate, epoch = decodeHyperparams(hyperparameters)
        return f", initEpsilon_{initEpsilon},trainEpsilon_{trainEpsilon},trainBatchSize_{trainBatchSize},trainMiniBatchSize_{trainMiniBatchSize},learningRate_{learningRate}, numPRCLLayers_{numPRCLLayers},numInternalLayers_{numInternalLayers}, activation_{activation}, lossfn_{lossFct}, epochs_{epoch}.pdf"
    else:
        numPRCLLayers, numInternalLayers, activation, initEpsilon, trainEpsilon, trainBatchSize, trainMiniBatchSize, lossFct, learningRate = decodeHyperparams(hyperparameters)
        return f", initEpsilon_{initEpsilon},trainEpsilon_{trainEpsilon},trainBatchSize_{trainBatchSize},trainMiniBatchSize_{trainMiniBatchSize},learningRate_{learningRate}, numPRCLLayers_{numPRCLLayers},numInternalLayers_{numInternalLayers}, activation_{activation}, lossfn_{lossFct}.pdf"
    
def plot_loss_eq2(EPOCHS,lossTrainEq, lossTrainNoEq, lossValidEq, lossValidNoEq,lossTrainEqErr, lossTrainNoEqErr, lossValidEqErr, lossValidNoEqErr, hyperparameters):
    
    W = 0.5
    H = 0.4
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
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
    fig.suptitle(f"SPACETIME SYMMETRY  Loss fn plots\n{plotTitle(hyperparameters)}",y=1.3)
    fig.savefig(f"SPACETIME SYMMETRY Loss fn plots{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 
    
    
def plot_loss_eq(lossTrainEq, lossTrainNoEq, lossValidEq, lossValidNoEq, hyperparameters):
    
    W = 0.5
    H = 0.4
    
    fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12, 5))
    #outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    
    # train data
    
    #inner = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0], wspace=W, hspace=H)
    #ax0 = plt.Subplot(fig, inner[0])
    #fig.add_subplot(ax0)
    
    
    ax0.plot(lossTrainEq, 'r', label = "Equivariance")
    ax0.plot(lossTrainNoEq, 'b', label = "No equivariance")
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss function')
    ax0.set_title('Train data')
    ax0.set_yscale('log')
    ax0.legend(loc="lower center")

    # valid data
    
    #ax1 = plt.Subplot(fig, inner[1])
    #fig.add_subplot(ax1)

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


def plot_loss(lossTrain, lossValid, hyperparameters):
    plt.plot(lossTrain, 'r', label = "Train Loss")
    plt.plot(lossValid, 'b', label = "Valid Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss function')
    plt.title(f"Loss function plot\n{plotTitle(hyperparameters)}", y=1.01)
    plt.yscale('log')
    plt.legend()
    savepath = f"Loss fn plot{pathname(hyperparameters)}"
    plt.savefig(savepath, bbox_inches='tight', dpi=150)
    plt.clf()
    plt.close()
    

class ACTION(torch.nn.Module):
    def __init__(self, Hubbard2SiteModel):
        super(ACTION,self).__init__()

        self.HM = Hubbard2SiteModel

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
    
    
def plot_actionStatistics(validData, NNvalidData, logDetValid, hyperparameters, HM):
    
    # get the values for the x and y axes
    
    S_NNvalidData, S_validData, diffSvalid, y1phi, y1NNphi, x1phi, x1NNphi, x2phi, x2NNphi, x3phi,\
                        x3NNphi, x4phi, x4NNphi, y3phi, y2phi, y2NNphi, y4phi = get_x_y_values(validData, NNvalidData, logDetValid, HM)
    
    # create the large figure where all the actions plots will be displayed
    
    W = 0.5
    H = 0.4

    fig = plt.figure(figsize=(12, 28))
    outer = gridspec.GridSpec(5, 1, wspace=0.2, hspace=0.2,height_ratios=[1,2,2,2,2])

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
    
    
def plot_fieldStatistics(validData, NNvalidData, logDetValid, hyperparameters, HM):
    
    # get the values for the x and y axes (almost all of them)
    
    S_NNvalidData, S_validData, diffSvalid, y1phi, y1NNphi, x1phi, x1NNphi, x2phi, x2NNphi, x3phi,\
                        x3NNphi, x4phi, x4NNphi, y3phi, y2phi, y2NNphi, y4phi = get_x_y_values(validData, NNvalidData, logDetValid, HM)
    
    # create the large figure where all the field plots will be displayed
    
    fig = plt.figure(figsize=(6, 10))
    outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.3)

    # i=0
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[0], wspace=0.1, hspace=0.2)
    ax0 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax0)

    # Re <phi> vs Im <phi>
    yphi = x1phi
    yNNphi = x1NNphi

    xphi = x2phi
    xNNphi = x2NNphi

    ax0.plot(xphi,yphi, '.r', label = 'phi')
    ax0.plot(xNNphi,yNNphi, '.b', label = 'NN(phi)')
    ax0.set_xlabel('Im <phi>')
    ax0.set_ylabel('Re <phi>')
    ax0.legend()

    # i=1
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    ax1 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax1)

    # mod Re phi vs Im mod phi

    yphi = x3phi
    yNNphi = x3NNphi

    xphi = x4phi
    xNNphi = x4NNphi
    ax1.scatter(xphi,yphi, c='r', marker='.',label = 'phi')
    ax1.scatter(xNNphi,yNNphi, c='b', marker='.',label = 'NN(phi)')
    ax1.set_xlabel('|Im phi|')
    ax1.set_ylabel('|Re phi|')
    ax1.legend()


    # i=2
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[2], wspace=0.1, hspace=0.1)
    ax2 = plt.Subplot(fig, inner[0])
    fig.add_subplot(ax2)

    # Re phi vs Im phi
    
    # get some new data
    yphi = validData.sum(dim=(-1,-2)).real.detach().numpy()
    yNNphi = NNvalidData.sum(dim=(-1,-2)).real.detach().numpy()

    xphi = validData.sum(dim=(-1,-2)).imag.detach().numpy()
    xNNphi = NNvalidData.sum(dim=(-1,-2)).imag.detach().numpy()
    
    #plot 
    
    ax2.scatter(xphi,yphi, c='r',marker='.',  label = 'phi')
    ax2.scatter(xNNphi,yNNphi, c='b', marker='.',label = 'NN(phi)')
    ax2.set_xlabel('Im phi')
    ax2.set_ylabel('Re phi')
    ax2.legend()
    fig.suptitle(f"Field statistics\n{plotTitle(hyperparameters)}",y=1.05)

    fig.savefig(f"Field statistics{pathname(hyperparameters)}",bbox_inches='tight', dpi=150)
    fig.clear()
    plt.close(fig) 