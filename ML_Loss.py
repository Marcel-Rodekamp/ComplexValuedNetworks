import torch
from Hubbard2SiteModel import Hubbard2SiteModel 

class Integrator(torch.nn.Module):
    def __init__(self, stepSize: float, HM: Hubbard2SiteModel):
        super(Integrator,self).__init__()

        # b-a/n = \tau_f - 0 / N
        self.stepSize = stepSize 

        self.f = lambda phi: HM.force(phi).conj()

    def forward(self,phiSteps: torch.tensor):
        r"""
            phiSteps: shape (N,[Nconf],Nt,Nx) with N support points 
        """
        fPhi = None
        if phiSteps.dim() == 3:
            fPhi = self.f(phiSteps)
        elif phiSteps.dim() == 4:
            fPhi = torch.zeros_like(phiSteps)

            for k in range(phiSteps.shape[0]):
                fPhi[k] = self.f(phiSteps[k])
        else:
            raise RuntimeError(f"Integrator not implemented for fields of dimension: {phi.shape}")

        # trapezoidal rule 
        integ = 0.5*(fPhi[0] + fPhi[-1])
        integ += fPhi[1:-1].sum(dim=0)
        return self.stepSize * integ

class IntegrateHolomorphicFlowLoss(torch.nn.Module):
    def __init__(self,  integrator: torch.nn.Module):
        super(IntegrateHolomorphicFlowLoss,self).__init__()

        self.integrator = integrator

    def forward(self, phiSteps: torch.tensor):
        r"""
            phiSteps: List of applied network
            
            phiSteps[0] = phi(\tau = 0)
            phiSteps[1] = phi(\tau =   \Delta \tau) = PRACL(phi(\tau=0))
            phiSteps[2] = phi(\tau = 2*\Delta \tau) = PRACL(PRACL(phi(\tau=0)))
            ...
            phiSteps[N-1]=phi(\tau = (N-1)*\Delta \tau = \tau_f) = PRACL( ... PRACL(phi(\tau=0)))
        """
        difference = phiSteps[phiSteps.shape[0]-1] - (self.integrator(phiSteps) + phiSteps[0])

        return difference.abs().mean()

