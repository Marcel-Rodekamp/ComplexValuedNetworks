import torch
import numpy as np
from lib_tensorSpecs import torchTensorArgs


class Hubbard2SiteModel:
    def __init__(self,Nt,beta,U,mu, tangentPlaneOffset = 0):
        self.Nt = Nt 
        self.Nx = 2
        self.beta = beta
        self.tangentPlaneOffset = tangentPlaneOffset
        self.delta = beta/self.Nt

        self.U = U*self.delta  
        self.mu = mu*self.delta

        self.particle_species = +1
        self.hole_species = -1
    
        # hopping matrix (particles)
        # exp( \kappa + C^p )
        self.expKappa_p = torch.zeros((self.Nx, self.Nx),**torchTensorArgs)
        self.expKappa_p[0, 0] = np.cosh(self.delta) * np.exp(self.mu)
        self.expKappa_p[0, 1] = np.sinh(self.delta) * np.exp(self.mu)
        self.expKappa_p[1, 0] = np.sinh(self.delta) * np.exp(self.mu)
        self.expKappa_p[1, 1] = np.cosh(self.delta) * np.exp(self.mu)
        
        # hopping matrix (holes)
        # exp( \kappa + C^h )
        self.expKappa_h = torch.zeros((self.Nx, self.Nx),**torchTensorArgs)
        self.expKappa_h[0, 0] =  np.cosh(self.delta) * np.exp(-self.mu)
        self.expKappa_h[0, 1] = -np.sinh(self.delta) * np.exp(-self.mu)
        self.expKappa_h[1, 0] = -np.sinh(self.delta) * np.exp(-self.mu)
        self.expKappa_h[1, 1] =  np.cosh(self.delta) * np.exp(-self.mu)

    def __repr__(self):
        return f"Hubbard2SiteModel(Nt={self.Nt},\u03B2={self.beta},U={self.U},\u03BC={self.mu},TP={self.tangentPlaneOffset:.2})"

    def M(self,phi,species):
        r"""
            ...
        """

        # create memory for constructing the fermion matrix 
        # torchTensorArgs are golbally defined in lib_tensorSpecs
        M = torch.zeros((self.Nt, self.Nx, self.Nt, self.Nx), **torchTensorArgs)

        # identify the hopping matrix according to the species
        if species == self.particle_species:
            expKappa = self.expKappa_p
        elif species == self.hole_species:
            expKappa = self.expKappa_h
        else:
            return ValueError(f"Fermion Matrix got wrong species argument. Must be +/- 1 but is {species}.")
        
        
        # precompue e^{+/- i phi}
        expphi = torch.exp(species*1.j*phi)
        
        # fill the t diagonal entries with identities in spatial space
        ts = torch.arange(0,self.Nt-1)
        M[ts,:,ts,:] = torch.eye(self.Nx,**torchTensorArgs)
        M[self.Nt - 1, :, self.Nt - 1, :] = torch.eye(self.Nx,**torchTensorArgs)
        
        # compute the bulk time slices 
        M[ts+1, :, ts, :] = -expKappa[:, :] * expphi[ts,None, :]
    
        # compute the boundary time slice
        M[0, :, self.Nt-1, :] = expKappa[:, :] * expphi[self.Nt-1,None, :]
    
        return M

    def logdetM(self,phi,species):    
        r"""
            ...
        """
        return torch.log(torch.linalg.det(self.M(phi,species).reshape(self.Nt*self.Nx,self.Nt*self.Nx)))

    def action(self,phi):
        r"""
            ...
        """

        # compute the action obtained by Hubbard Statonovich Transformation
        return (phi*phi).sum()/(2*self.U) \
                - self.logdetM(phi,species = self.particle_species) \
                - self.logdetM(phi,species = self.hole_species) 

