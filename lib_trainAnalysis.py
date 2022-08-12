import torch
import matplotlib.pyplot as plt

def averagePhi(phi):
    r"""
        This computes the avarage of the configuration over the space time 
        volumen 

        \f[
            \frac{1}{V} \sum_{t,x} \phi_{t,x}
        \f] 
    """
    return phi.sum(dim=(-1,-2))/(Nt*Nx)

def modPhi(phi): 
    r"""
        This computes the L2 norm of the configuration over the space time
        volume

        \f[
            |\phi| = \sqrt{ \sum_{t,x} \phi_{t,x}^2 }
        \f]
    """
    return torch.sqrt((torch.square(phi)).sum(dim=(-1,-2)))
        

def plot_loss( lossTrain, lossValid, savepath = "loss.pdf"):
    plt.plot(lossTrain, 'x', label = "Train Loss")
    plt.plot(lossValid, 'x', label = "Valid Loss")
    plt.yscale('log')
    plt.legend()
    plt.savefig(savepath)
    plt.clf()
    plt.close()
