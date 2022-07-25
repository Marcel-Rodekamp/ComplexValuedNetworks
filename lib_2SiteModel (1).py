import torch
import numpy as np
from lib_tensorSpecs import torchTensorArgs


class Hubbard2SiteModel:
    r"""
        This class contains everything that characterises the Hubbard Model for the 2 site problem.
    """
    def __init__(self,Nt,beta,U,mu, tangentPlaneOffset = 0):
        r"""
             \param: Nt, int
                    - Number of time slices
            \param: beta, int
                    - we control the continuum limit with it
            \param: U, int
                    - On site interaction
            \param: mu, int
                    - Chemical potential
            \param: tangentPlaneOffset, float
                    - position of the tangent plane for the particular combination of (beta, U, mu). Default: 0
        """
        # Number of ions in the lattice - it will always be 2 in this project
        self.Nx = 2
        
        self.Nt = Nt 
        self.beta = beta
        self.tangentPlaneOffset = tangentPlaneOffset
        
        # Lattice spacing
        self.delta = beta/self.Nt

        # Make all variables unit less
        self.U = U*self.delta  
        self.mu = mu*self.delta

        # Identify the two different species by +/-1
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
             \params:
                - phi: torch.tensor(Nt,Nx), configuration
                - species: +1 for particles, -1 for holes
        
            M(phi)_{t',x'; t,x} = \delta_{t',t} \delta_{x',x} - exp(s*(\Kappa+\mu))_{x',x} exp(i*s*\Phi_{t,x}) \delta_{t',t+1} 
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
            \params:
                - phi: torch.tensor(Nt,Nx), configuration
                - species: +1 for particles, -1 for holes
        
            \log \det [M(phi)_{t',x'; t,x}] 
        """
        return torch.log(torch.linalg.det(self.M(phi,species).reshape(self.Nt*self.Nx,self.Nt*self.Nx)))

    def action(self,phi):
        r"""
            \params:
                - phi: torch.tensor(Nt,Nx), configuration
                - species: +1 for particles, -1 for holes
        
            S(\phi) = \frac{1}{2U} \phi^T \phi - \log{\det{ M^p \cdot M^h }}   
        """

        # compute the action obtained by Hubbard Stratonovich Transformation
        return (phi*phi).sum()/(2*self.U) \
                - self.logdetM(phi,species = self.particle_species) \
                - self.logdetM(phi,species = self.hole_species) 

    def calculate_batch_action(self, batch_phi):
        r"""
            \params:
                - batch_phi: torch.tensor(batchSize,Nt,Nx), configurations
            Calculates and return the value of the action for each configuration from a batch containing a number of batchSize configurations.
        """

        batchSize, _, _ = batch_phi.shape 
        
        S = torch.zeros(batchSize,**torchTensorArgs)
        
        for batchID in range(batchSize):
            S[batchID] = self.action(batch_phi[batchID,:,:])
    
        return S
