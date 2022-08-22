import torch
from lib_2SiteModel import Hubbard2SiteModel

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


def singleStep(stepSize: float, phi: torch.tensor, HM: Hubbard2SiteModel):
    r""" 
        We apply a single RK4 step to the data and compare to the prediction
        Should we precomputed 
    """
    # f = (\frac{d S(phi)}{d phi} )^*
    k1 = HM.force(phi                       ).conj()
    k2 = HM.force(phi+(stepSize/2) * k1).conj()
    k3 = HM.force(phi+(stepSize/2) * k2).conj()
    k4 = HM.force(phi+ stepSize    * k3).conj()

    return phi + stepSize/6 * (k1+2*k2+2*k3+k4) 


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


if __name__ == "__main__":
    Nt = 8
    Nx = 2


    w = torch.rand((Nt,Nx), dtype = torch.cdouble, requires_grad = True)
    def NN(phi):
        return w*phi

    HM = Hubbard2SiteModel(
        Nt = Nt,
        beta = 4,
        U = 3,
        mu = 4,
        tangentPlaneOffset = 0
    )

    # Generate Train Data
    phi = torch.rand((Nt,Nx), dtype = torch.cdouble).real + 0j
    phiDT = singleStep(stepSize = 0.1, phi = phi, HM = HM)
    # maybe save it for later reuse?

    # Single Step Training:
    preLoss = torch.nn.L1Loss(reduction='mean')
    NNPhi = NN(phi)
    loss = preLoss(phiDT,NNPhi)
    print(loss) 
    # call backward...


    # Generate Train Data
    phi = torch.rand((Nt,Nx), dtype = torch.cdouble).real + 0j
    # maybe save it for later reuse?

    # Multi Step Training 
    integrator = Integrator(stepSize = 0.1, HM = HM)
    lossFct = IntegrateHolomorphicFlowLoss(integrator = integrator)

    # some general network arch. around this:
    def flow(phi, N):
        phiSteps = [phi]
        for n in range(1,N):
            phiSteps.append( NN(phiSteps[n-1]) )
        return torch.stack( tuple(phiSteps),dim = 0)

    pred = flow(phi,3)
    loss = lossFct( pred )
    print(loss)

    # Basic network idea:
    # \Phi(\Delta\tau) = PRACL(\Phi(0))
    # \Phi_k = \Phi( k \Delta\tau ) 
    # with network:
    # \Phi_0 = \Phi(0)
    # \Phi_1 = PRACL(\Phi_0)
    # ...
    # \Phi_k = PRACL(\Phi_{k-1})
    # ...
    # \Phi_N = PRACL(\Phi_{N-1})

    # Basic Training idea:
    # 1. Single Step Training Phi -> L1(NN(Phi), RK4(Phi))
    #       - Nconf large (only 4 forces required)
    #       - Nepoch very large (no forces required)
    #       - learning rate very small 
    # 2. Multi Step Training Phi  -> IntegrateHolomorphicFlowLoss(NN(Phi))
    #       - Nconf small to medium (N*Nepoch forces required)
    #       - Nepoch small (50 - 500) (N*Nepoch forces required)
    
    # Oscillating Training idea
    # 1. Single Step Training 
    # 2. Multi Step Training 
    # 3. Single Step Training
    # 4. Multi Step Training 
    # 5. ...
    #       - Nepoch for Multi step must be small depending on the number 
    #         of repetitions

    # Having another single step training closer to the thimble 
    # 1. Single Step Training 
    # 2. Multi Step Training
    # 3. Phi -> NN(...(NN(Phi))) -> RK4(NN(...NN(Phi))) = psi
    # 4. Single Step Training (psi,NN(psi))
    # 5. Multi Step Training






