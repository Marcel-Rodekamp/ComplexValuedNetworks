import torch 
from Hubbard2SiteModel import Hubbard2SiteModel

def singleRK4Step(stepSize: float, phi: torch.tensor, HM: Hubbard2SiteModel):
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
