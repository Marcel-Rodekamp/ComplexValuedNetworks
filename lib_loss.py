import torch

class StatisticalPowerLoss(torch.nn.Module):
    
    # the loss function that maximises the statistical power

    def __init__(self, Hubbard2SiteModelIsleIsleAction, partitionFunctionEstimate = 1):
        
        r"""
            \param: Hubbard2SiteModel, class
            \param: partitionFunctionEstimate, int
        """
        super(StatisticalPowerLoss,self).__init__()

        self.HM = Hubbard2SiteModelIsleIsleAction

        # register the estimate of the partition function 
        self.register_buffer(
            name = 'Z',
            tensor = torch.tensor([partitionFunctionEstimate])
        )

    def forward(self, phi, logDetJ_NN):
        
        r"""
            \param: phi = torch.tensor(batchSize, Nt, Nx)
            \param: logDetJ_NN, torch.tensor(batchSize)
        """
        # the effective action is the action minus the log det of the NN
        Seff = self.HM.calculate_batch_action(phi) - logDetJ_NN
        
        # the normalised statistical power:
        # \sigma = 
        
        return torch.exp(-Seff).abs().mean()/self.Z


class MinimizeImaginaryPartLoss(torch.nn.Module):
    
    # the loss function that minimises the imaginary part of the difference between the action and the log det of the NN
    
    def __init__(self, Hubbard2SiteModelIsleIsleAction):
        super(MinimizeImaginaryPartLoss,self).__init__()

        self.HM = Hubbard2SiteModelIsleIsleAction

    def forward(self,phi,logDetJ_NN):
        Seff = self.HM.calculate_batch_action(phi) - logDetJ_NN
        with torch.no_grad():
            error = (Seff.imag).abs().var()
        return (Seff.imag).abs().mean(), error

