import torch
import numpy as np
from lib_tensorSpecs import torchTensorArgs

try:
    import isle 
    import isle.action
except:
    print("No isle support found.")



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
                S[n] = ctx.action.eval(isle.CDVector(phi[n,:,:].detach().reshape(Nt*Nx).numpy()))

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

        self.S = Action(Nt,beta,U,mu,lattice='two_sites')

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
        return self.S(phi)

    # The force term from log det M^p
    def TrMinvM(self,phi,species):
        r"""
            \params:
                - phi: torch.tensor(Nt,Nx), configuration
                - species: +1 for particles, -1 for holes
            
            \frac{d \log{\det{ M(\phi) }}}{ d\phi } = \Tr{ M^{-1}(\phi) \cdot \frac{dM(\phi)}{d\phi} }
            
            where 
            
            \frac{dM(\phi)_{t',x'; t,x}} }{d\phi_{t,x}} = -i*s* exp(s*(\Kappa+\mu))_{x',x} exp(i*s*\phi_{t,x}) \delta_{t',t+1} 
        """
        # Tr = d logdetM(\phi)/d\phi_{t,x}
        Tr = torch.zeros((self.Nt,self.Nx),**torchTensorArgs)
        
        # determine the expression of exp(s*Kappa) from the species 
        if species == self.particle_species:
            expKappa = self.expKappa_p
        elif species == self.hole_species:
            expKappa = self.expKappa_h
        else:
            return ValueError(f"Force got wrong species argument. Must be +/- 1 but is {species}.")
        
        expphi = torch.exp(species*1j*phi)
        
        # again reshape M for torch to understand it as a matrix
        Minv = torch.linalg.inv(
            self.M(phi,species).reshape(self.Nt * self.Nx, self.Nt * self.Nx)
        ).reshape(self.Nt, self.Nx, self.Nt, self.Nx)
        
        ts = torch.arange(0,self.Nt-1)
        
        #bulk time slices
        temp = torch.diagonal(torch.tensordot(Minv[ts,:,ts+1,:],expKappa[:,:],dims=1), dim1 = 1, dim2 = -1)
            
        # (TrMinvM)_{t,x} = -i * s * temp_{t,x} * exp(i*s*\phi_{t,x})
        Tr[ts, :] = -1.j * species * temp[:] * expphi[ts, :]
        
        # boundary time slice
        # term t = self.Nt=0, t' = Nt=-1 
        #for x in range(Nx):
            # temp_{Nt-1,x} = M^{-1}_{0,x; t'=0,x'} exp(s*(\Kappa+\mu))_{x',x} \delta_{t'=0,t+1}tmp_{t,x} 
            #               = M^{-1}_{t,x,t+1} @ exp(Kappa) 
        temp = torch.diag(torch.tensordot(Minv[self.Nt-1,:,0,:],expKappa[:,:],dims=1))

        # (TrMinvM)_{Nt-1,x} = i * s * tmp_{t,x} * exp(i*s*phi_{t,x})
        Tr[self.Nt - 1, :] = 1.j * species * temp * expphi[self.Nt-1, :]
    
        return Tr        
        
    def force(self, phi):
        r"""
            \params:
                - phi: torch.tensor(Nt,Nx), configuration
                
            F(phi) = - dS(phi)/dphi = - d/dphi ( 1/2U phi^2 - log det M^p - log det M^h )
                                    = - [ 1/U phi - Tr( M^{p,-1} dM^p/dphi + M^{h,-1} dM^h/dphi ) ]
        """
        if phi.dim() == 2:
            return (phi/self.U - self.TrMinvM(phi,+1) - self.TrMinvM(phi,-1))
        elif phi.dim() == 3:
            Nconf,_,_ = phi.shape 
            f = torch.zeros_like(phi)
            
            for n in range(Nconf):
                f[n] = -self.TrMinvM(phi[n],+1) - self.TrMinvM(phi[n],-1)

            f += phi/self.U 

            return f 
        else:
            raise RuntimeError(f"force not implemented for phi of shape {phi.shape}")

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


if __name__ == "__main__":
    HM = Hubbard2SiteModel(
        Nt = 4,
        beta = 4,
        U = 3,
        mu = 4,
        tangentPlaneOffset = 0
    )

    phi = torch.rand((2,4,2), dtype = torch.cdouble, requires_grad = True).real + 0j
    print(f" S = {HM.action(phi)}")
    print(f" F = {HM.force(phi)}")

