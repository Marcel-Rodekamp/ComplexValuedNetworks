import torch

class AffineCoupling(torch.nn.Module):
    r"""
    This implements the affine coupling.
    \f[
        ACL(x_A,x_B) = exp( m(x_B) ) * x_A + a(x_B)
    \f]

    Notice, this class is intended to be used in combination with the
    coupling layer defined below!
    """

    def __init__(self, m, a):
        r"""
            \param: m, torch.nn.Module, function multiplied to the input `xA`
                - Intended use: Neural Networks carrying the weights of the system
            \param: a, torch.nn.Module, function added to the input `xA`
                - Intended use: Neural Networks carrying the weights of the system
        """
        super(AffineCoupling, self).__init__()

        super().add_module("m", m)
        super().add_module("a", a)

    def forward(self, xA, xB):
        r"""
            Computes the forward pass of the coupling:
            \f[
                ACL(x_A,x_B) = exp( m(x_B) ) * x_A + a(x_B)
            \f]
            and the log det Jacobian in respect to the `xA` components.
            The jacobian is ment to be used in combination with the coupling
            layer defined further below. It does NOT consider the xB components
            and only returns the (diagonal) values required for taking the
            formal determinant.
            \f[
                J_{x_A} = \frac{\partial ACL(x_A,x_B)}{\partial x_A}
                        = diag[exp(m(x_B))]
            \f]
            The calculation of the log det J does not introduce gradients
            for training the Network!
        """
        mout = self.m(xB)
        aout = self.a(xB)

        # returns: forward value, diagonal elements of the xA jacobian
        
        return mout.exp() * xA + aout, mout.detach().sum(-1).sum(-1)

class PRCL(torch.nn.Module):
    def __init__(self, Nt: int, Nx: int, coupling: torch.nn.Module ):
        super(PRCL, self).__init__()
        super().add_module(f"coupling", coupling)

        self.mask = torch.ones(Nt, Nx, dtype=torch.bool, requires_grad=False)
        # Set every second spatial site (A,B) to false 
        self.mask[:, ::2] = False

    def forward(self, input_phi):
        A = input_phi[..., self.mask].unsqueeze(dim=-1)
        B = input_phi[..., ~self.mask].unsqueeze(dim=-1)
        
        A, logDetJc1 = self.coupling(A, B)
        
        B, logDetJc2 = self.coupling(B, A)

        out = torch.empty_like(input_phi)
        out[..., self.mask] = A.squeeze(dim=-1)
        out[..., ~self.mask] = B.squeeze(dim=-1)

        return out, logDetJc1 + logDetJc2

class Equivariance(torch.nn.Module):
    def __init__(self,NN):
        
        # initialise the class
        super(Equivariance,self).__init__()
        
        # add the NN as a module
        self.add_module("NN",NN)

    def forward(self,phi):
        r"""
            \param: phi: batch of configurations
            The forward pass will implement the TL layer, using the NN
        """
        
        # get the dimensions of phi
        if phi.dim() == 3:
            Nconf,_,_ = phi.shape
            
            # get the time slice t0 where each config has a minimum value along x=0
            t = phi.abs().argmin(dim=1)
            
            T0 = t[:,0]
       
            # loop through all the configurations in the batch and roll them
            # (the 'for' loop is needed because 'shifts' can only take int as argument)
            for i in range(Nconf):
                
                # space translation
                if t[i][0].item() > t[i][1].item():
                    T0[i] = t[i][1]
                    phi[i,...].flip(1)
                                
                
                # translate the whole configuration in time by t0
                phi[i,...].roll(shifts = T0[i].item(), dims=0)
                
               
                
            # apply the NN
            NNphi, logDet = self.NN(phi)
            
            # invert the time translation, rolling back with -t0
            for i in range(Nconf):
                NNphi[i,...].roll(shifts = -T0[i].item(), dims=0)
                if t[i][0].item() > t[i][1].item():
                    NNphi[i,...].flip(1)
            
            # this whole class is supposed to be a new NN, so again we return the resulting configurations and the logDet
            # the logDet is unchanged by the time translation, so it is just the logDet returned by the initial NN
            return NNphi, logDet
        elif phi.dim() == 2:
            # get the time slice t0 where each config has a minimum value along x=0
            t = phi.abs().argmin(dim=0)
            T0 = t[0]
            # space translation
            if t[0].item() > t[1].item():
                T0 = t[1]
                phi.flip(1)
                                
            # translate the whole configuration in time by t0
            phi.roll(shifts = T0.item(), dims=0)

            # apply the network
            NNphi, logDet = self.NN(phi)

            # invert the time translation, rolling back with -t0
            NNphi.roll(shifts = -T0.item(), dims=0)
            if t[0].item() > t[1].item():
                NNphi.flip(1)
            
            return NNphi, logDet

        else:
            raise RuntimeError ("Equivariance expects input of dimension 3 (Nconf,Nt,Nx) or dimension 2 (Nt,Nx)")
 
class Sequential(torch.nn.Module):
    def __init__(self, model_list):
        super(Sequential, self).__init__()

        self.module_list = torch.nn.ModuleList(model_list)
        self.len = len(model_list)

    def forward(self, input_phi):
        out = input_phi

        logDetJ = None
        if input_phi.dim() == 3:
            Nconf, _,_ = input_phi.shape
            logDetJ = torch.zeros(Nconf, dtype = input_phi.dtype, device = input_phi.device)
        else:
            logDetJ = 0
        
        for m in self.module_list:
            out,ldj = m(out)
            logDetJ += ldj

        return out, logDetJ

    def __getitem__(self, item):
        return self.module_list[item]

class Flow(torch.nn.Module):
    r"""
        Given a neural network perfoming a single step in holomorphic flow 
        we can reapply it N times to get to a desired flow time.
        This class handles the number of steps N taken and ensures the correct
        setup for training mode (all steps 0 - (N-1) are returned) and 
        evaluation mode (only the last step is returned)
    """
    def __init__(self,NN: torch.nn.Module, N: int):
        # initialise the class
        super(Flow,self).__init__()
       
        # add the NN as a module
        self.add_module("NN",NN)
        self.N = N
        
    def __training_forward(self, phi):
        # Prepare a list of steps to get all interpediate steps (used as 
        # support points in the loss function)
        phiSteps = [None] * self.N
        phiSteps[0] = phi

        logDetFlow = 0
        
        for n in range(1,self.N):
            Psi, logDet = self.NN(phiSteps[n-1])
            phiSteps[n] = Psi
            logDetFlow += logDet

        # merge the different steps in a single torch tensor for easier processing later
        return torch.stack( tuple(phiSteps),dim = 0), logDetFlow

    def __evaluation_forward(self,phi):
        logDetFlow = 0
        out = phi
        for n in range(1,self.N):
            out, logDet = self.NN(out)
            logDetFlow += logDet
         
        return out, logDetFlow

    def forward(self, phi): 
        if self.training:
            return self.__training_forward(phi)
        else:
            return self.__evaluation_forward(phi)

class LinearTransformation(torch.nn.Module):
    def __init__(self, Nt, Nx):
        r"""
            \param:
                - Nt: int, number of time slices compate Hubbard2SiteModel.Nt
                - Nx: int, number of ions compate Hubbard2SiteModel.Nx

            Initilizes a Linear transformation according to the 2 dimensional
            configuration phi ~ (Nt,Nx) used in the 2 Site Hubbard model.

            The transformation is defined by
            \f[
                \phi'_{t',x'} = \sum_{\t,x} \omega_{t',x',t,x} \phi_{t,x} + b_{t',x'}
            \f]
        """
        super(LinearTransformation, self).__init__()

        self.register_parameter(
            name  = 'bias', 
            param = torch.nn.Parameter(
                        torch.rand((Nt,Nx),dtype=torch.cdouble)
            )
        )
        self.register_parameter(
            name  = 'weight', 
            param = torch.nn.Parameter(
                torch.rand((Nt,Nx,Nt,Nx),dtype=torch.cdouble)
            )
        )

    def forward(self, x):
        return torch.tensordot(x,self.weight,dims=([-1,-2],[0,1])) + self.bias

