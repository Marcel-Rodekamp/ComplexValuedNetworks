import torch
import numpy as np
from lib_tensorSpecs import torchTensorArgs

try:
    import isle 
    import isle.action
except:
    print("No isle support found.")

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

    def calculate_batch_action(self, batch_phi):
        r"""
            ...
        """

        batchSize, _, _ = batch_phi.shape 
        
        S = torch.zeros(batchSize,**torchTensorArgs)
        
        for batchID in range(batchSize):
            S[batchID] = self.action(batch_phi[batchID,:,:])
    
        return S

class ActionImpl(torch.autograd.Function):

    @staticmethod
    def forward(ctx, phi, action):
        ctx.save_for_backward(phi)
        ctx.action = action
        
        # we will return S as the torch.tensor of action(s)
        S = None 

        # compute a batch of configs
        if len(phi.shape) > 2:
            Nconf,Nt,Nx = phi.shape 
            S = torch.zeros(Nconf,dtype=phi.dtype,device=phi.device,requires_grad=True)
            for n in range(Nconf):
                S[n] = ctx.action.eval(phi[n,:,:].detach().reshape(Nt*Nx).numpy())

        # compute a single config
        else:
            Nt,Nx = phi.shape 
            S = torch.tensor(
                ctx.action.eval(isle.CDVector(phi.detach().reshape(Nt*Nx).numpy())), 
                dtype=phi.dtype, 
                device=phi.device, 
                requires_grad=True
            )

        return S

    @staticmethod
    def backward(ctx,grad_output):
        phi, = ctx.saved_tensors

        out = torch.zeros_like(phi)
        
        # ok := boolean that check whether the matrix has become singular (True) or not (False)
        # default: False
        ok = False
        # compute backward for a batch of configs
        if len(phi.shape) > 2:
            Nconf,Nt,Nx = phi.shape 

            for n in range(Nconf):
                #try:
                force = -ctx.action.force(isle.CDVector(phi[n,:,:].detach().reshape(Nt*Nx).numpy()))

                out[n,:,:] = grad_output[n]*torch.from_numpy(
                    np.array(force)
                ).reshape(Nt,Nx).conj()
                #except:
                    #print("Singular matrix at this set of hyperparameters:")
                    #ok = True
                   # break

        # compute backward for a singe config
        else:
            #try: 
            Nt,Nx = phi.shape

            force = -ctx.action.force(isle.CDVector(phi.detach().reshape(Nt*Nx).numpy()))

            out[:,:] = grad_output*torch.from_numpy(
                np.array(force)
            ).reshape(Nt,Nx).conj()
            #except:
              #  print("Singular matrix at this set of hyperparameters:")
               # ok = True
               # break
        return out,None


class Action(torch.nn.Module):
    def __init__(self,Nt,beta,U,mu,lattice="two_sites"):
        super().__init__()
        self.lattice = isle.LATTICES[lattice]
        self.lattice.nt(Nt)

        self.params = isle.util.parameters(
            beta      = beta,                                   # inverse temperatur
            U         = U,                                      # On site interaction 
            mu        = mu,                                     # chemical potential
            sigmaKappa= -1,                                     # prefactor of kappa for holes/spin down
            hopping   = isle.action.HFAHopping.EXP,             # Hopping matrix 
            basis     = isle.action.HFABasis.PARTICLE_HOLE,     # Basis of the creation operators
            algorithm = isle.action.HFAAlgorithm.DIRECT_SINGLE, # Algorithm to compute log det M
        )

        self.action = isle.action.HubbardGaugeAction(self.params.tilde("U",self.lattice)) \
                    + isle.action.makeHubbardFermiAction(
                        self.lattice,
                        self.params.beta,
                        self.params.tilde("mu",self.lattice),
                        self.params.sigmaKappa,
                        self.params.hopping,
                        self.params.basis,
                        self.params.algorithm
                    )

        self.actionImpl = ActionImpl.apply

    def forward(self,phi):
        return self.actionImpl(phi,self.action)

class Hubbard2SiteModelIsleIsleAction:
    def __init__(self,Nt,beta,U,mu, tangentPlaneOffset = 0):
        self.Nt = Nt 
        self.Nx = 2
        self.beta = beta
        self.tangentPlaneOffset = tangentPlaneOffset
        self.delta = beta/self.Nt

        self.U = U*self.delta  
        self.mu = mu*self.delta

        self.actionModule = Action(Nt,beta,U,mu)

    def __repr__(self):
        return f"Hubbard2SiteModel(Nt={self.Nt},\u03B2={self.beta},U={self.U},\u03BC={self.mu},TP={self.tangentPlaneOffset:.2})"

    def action(self,phi):
        r"""
            ...
        """
        return self.actionModule(phi)

    def calculate_batch_action(self, batch_phi):
        r"""
            Implemented in actionModule already :)
        """

        return self.actionModule(batch_phi)