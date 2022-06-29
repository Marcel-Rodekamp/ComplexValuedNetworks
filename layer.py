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
    r"""
        This implements the paired affine coupling.
        \f[
            PRCL(x) = \begin{pmatrix}
                c_1(x_A,x_B) \\
                c_2(x_B,c_1(x_A,x_B))
            \end{pmatrix}
        \f]
        It automatically splits the input vector x randomly.
    """

    def __init__(self, Nt, Nx, coupling1, coupling2):
        r"""
            \param: V, int, expected size of the input dimension which is
                            divided int the A and B partitions.
            \param: coupling1, torch.nn.Module, the first coupling that is
                            applied on \f$x_A\f$ and \f$ x_B \f$.
                - Intended use: Couplings like the AffineCoupling above
            \param: coupling2, torch.nn.Module, the second coupling that is
                            applied on \f$x_B\f$ and \f$ c_1(x_A,x_B) \f$.
                - Intended use: Couplings like the AffineCoupling above
        """
        super(PRCL, self).__init__()

        super().add_module(f"c1", coupling1)
        super().add_module(f"c2", coupling2)

        # create a mask to seperate the input into the A and B partitions
        self.mask = torch.ones(Nt, Nx, dtype=torch.bool, requires_grad=False)
        # half of the input is passed (the true values)
        # the other half is not passed (the false values)
        self.mask[:, ::2] = False
        # shuffle the true and false values
        # self.mask[:,:] = self.mask[torch.randperm(Nt, Nx)]

    def forward(self, x):
        r"""
            \param: x, torch.tensor, input tensor

            This function computes the forward pass
            \f[
                PRCL(x) = \begin{pmatrix}
                    c_1(x_A,x_B) \\
                    c_2(x_B,c_1(x_A,x_B))
                \end{pmatrix}
            \f]
            and the associated log det Jacobian
            \f[
                J = J_{c_1(x_A)} * J_{c_2(x_B)}
            \f]
            This jacobian works in the real case and in the complex case if
            the couplings \f$ c_1,c_2 \f$ are holomorphic in A components.

        """
        # x -> (x_A,x_B)
        
        #print("coupling 1 is ", self.c1)
        
        #print("x is ", x)
        
        A = x[..., self.mask].unsqueeze(dim=-1)
        B = x[..., ~self.mask].unsqueeze(dim=-1)
        
        #print("A initial is ", A)
        #print("B initial is ", B)


        # x_A' = c_1(x_A,x_B)
        A, logDetJc1 = self.c1(A, B)
        #print("A final is ",A)


        #print("coupling 2 is ", self.c2)

        
        # x_B' = c_2( x_B,c_1(x_A,x_B) )
        B, logDetJc2 = self.c2(B, A)
        #print("B final is ",B)


        # x_out = (x_A',x_B')
        out = torch.empty_like(x)
        
        #print("out is ",out)
        out[..., self.mask] = A.squeeze(dim=-1)
        out[..., ~self.mask] = B.squeeze(dim=-1)

        return out, logDetJc1 + logDetJc2


class Sequential(torch.nn.Module):
    r"""
        This is just a wrapper of the Sequential container, that propagates
        the log det Jacobians returned from modules defined in this file.
        Based on the chain rule:
        \f[
            \log\det J = \sum_{l=0}^{N_L-1} \log\det J_l
        \f]
    """

    def __init__(self, model_list):
        r"""
            \param: model_list, list, list of neural networks stacked after each other
        """
        super(Sequential, self).__init__()

        self.module_list = torch.nn.ModuleList(model_list)
        self.len = len(model_list)

    def forward(self, x):
        r"""
            \param: x, torch.tensor, input Tensor

            Computes the forward pass of a neural network with the form
            \f[
                NN(x) = \left(f_{N_L-1} \circ f_{N_L-2} \circ \cdots \circ f_{0}\right)(x)
            \f]
        """
        logDetJ = 0
        for m in self.module_list:
            x, logDetJl = m(x)
            logDetJ += logDetJl

        return x, logDetJ

    def __getitem__(self, item):
        return self.module_list[item]


# =======================================================================
# For convenience I define a bunch of factory functions:
# =======================================================================

def createPRCL(V, Nlayer, couplingFactory, **kwargs):
    r"""
        \param: V, int, intended volume
        \param: Nlayer, int, number of paired coupling layers
        \param: couplingFactory, function, create a coupling function model
                    carying the weights of the neural network
        \param: kwargs, key word arguments, arguments passed to the coupling factory

        This function creates a paired coupling layer with coupling functions
        composed by couplingFactory. Here it is assumed that the couplings
        m and a are of the same type but with different weights.
    """

    moduleList = []
    for l in range(Nlayer):
        moduleList.append(
            PRCL(V, couplingFactory(**kwargs), couplingFactory(**kwargs))
        )

    return Sequential(moduleList)


def createACL(mFactory, aFactory, **kwargs):
    r"""
        \param: mFactory, function, function that creates a multiplicative
                    coupling partner of an ACL. Must be argumentless!
        \param: mFactory, function, function that creates a additive
                    coupling partner of an ACL. Must be argumentless!
        \param: **kwargs, keyword arguments, passed to BOTH factories

        A simple wrapper creating an AffineCoupling given factory functions
        to create the multiplicative and additive networks.
    """

    return AffineCoupling(mFactory(**kwargs), aFactory(**kwargs))


def createACLShared(factory, *args, **kwargs):
    r"""
        \param: factory, function, function that creates a neural network.
        \param: *args, arguments, passed to factory
        \param: **kwargs, keyword arguments, passed to factory

        A simple wrapper creating an AffineCoupling. It assumes parameter
        shareing for the multiplicative and additive network i.e.

        m = t = factory(*args,**kwargs)

        Note: I didn't test this before but other groups use the network in
        this way so we might want to try.
        """

    NN = factory(*args, **kwargs)

    return AffineCoupling(NN, NN)
