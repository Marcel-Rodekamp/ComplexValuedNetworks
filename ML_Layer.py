import torch

class Coupling(torch.nn.Module):
    def __init__(self, NN: torch.nn.Module, tangentPlaneOffset: float):
        super(Coupling, self).__init__()
        super().add_module("NN", NN)

        self.tangentPlaneOffset = tangentPlaneOffset

    def forward(self, input_A, input_B):
        out = input_A.real + 1j * self.NN(input_B.real) + 1j* self.tangentPlaneOffset

        if input_A.dim() == 3:
            Nconf, _, _ = input_A.shape
            return out, torch.zeros(Nconf, dtype = input_A.dtype, device = input_A.device)
        else:
            return out, torch.zeros(1, dtype = input_A.dtype, device = input_A.device)

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

class RealLinearTransformation(torch.nn.Module):
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
        super(RealLinearTransformation, self).__init__()

        self.register_parameter(
            name  = 'bias', 
            param = torch.nn.Parameter(
                        torch.rand((Nt,Nx))
            )
        )
        self.register_parameter(
            name  = 'weight', 
            param = torch.nn.Parameter(
                torch.rand((Nt,Nx,Nt,Nx))
            )
        )

    def forward(self, x):
        return torch.tensordot(x,self.weight,dims=([-1,-2],[0,1])) + self.bias

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

class Id(torch.nn.Module):
    def __init__(self):
        super(Id, self).__init__()
        self.I = torch.nn.Identity()

    def forward(self,*args):
        return self.I(*args)

    def calcLogDetJ(self,*args,**kwargs):
        return 0

if __name__ == "__main__":

    def modPi(S):
        from numpy import pi
        return (S + pi) % (2 * pi) - pi

    V = 32
    def create_coupling():

        lin1 = torch.nn.Linear(V//2,V//2).to(torch.cdouble)
        torch.nn.init.eye_(lin1.weight)
        torch.nn.init.zeros_(lin1.bias)
        lin2 = torch.nn.Linear(V//2,V//2).to(torch.cdouble)
        torch.nn.init.eye_(lin2.weight)
        torch.nn.init.zeros_(lin2.bias)
        return torch.nn.Sequential(
            lin1,
            torch.nn.Tanh(),
            lin2
        )

    def create_PRACL():
        return  PRCL(V,
                coupling1= AffineCoupling(
                    m=create_coupling(),
                    a=create_coupling()
                ),
                coupling2= AffineCoupling(
                    m=create_coupling(),
                    a=create_coupling()
                )
        )

    model = create_PRACL()

    #print(f"Test worked: {model.test(torch.randn(size=(100,V),dtype=torch.cdouble))}")

    phi_input = torch.randn(V,dtype=torch.cdouble)
    # phi_input = torch.ones(V,dtype=torch.cdouble)
    phi_out = model(phi_input)

    logDet_exa = model.calcLogDetJ(phi_input)
    logDet_num = torch.autograd.functional.jacobian(
        model, phi_input
    ).det().log().conj()

    logDet_num.imag = modPi(logDet_num.imag)
    logDet_exa.imag = modPi(logDet_exa.imag)

    res_r = torch.abs(logDet_exa.real - logDet_num.real)
    res_i = torch.abs(logDet_exa.imag - logDet_num.imag)
    print(f"logDet equals: {res_r < 1e-14 and res_i < 1e-14}")
    print(f"res_r = {res_r}, res_i = {res_i}")
    print(f"logDet numerical : {logDet_num:.16e}")
    print(f"logDet analytical: {logDet_exa:.16e}")
    # print(logDet_exa,logDet_num)

