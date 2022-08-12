import torch

class StatisticalPowerLoss(torch.nn.Module):

    def __init__(self, Hubbard2SiteModel, partitionFunctionEstimate = 1):
        super(StatisticalPowerLoss,self).__init__()

        self.HM = Hubbard2SiteModel

        self.register_buffer(
            name = 'Z',
            tensor = torch.tensor([partitionFunctionEstimate])
        )

    def forward(self, phi, logDetJ_NN):
        Seff = self.HM.calculate_batch_action(phi) - logDetJ_NN
        return torch.exp(-Seff).abs().mean()/self.Z


class MinimizeImaginaryPartLoss(torch.nn.Module):
    def __init__(self, Hubbard2SiteModel):
        super(MinimizeImaginaryPartLoss,self).__init__()

        self.HM = Hubbard2SiteModel

    def forward(self,phi,logDetJ_NN):
        Seff = self.HM.calculate_batch_action(phi) - logDetJ_NN
        return (Seff.imag).abs().mean()

