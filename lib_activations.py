import torch

class complexRelu(torch.nn.Module):
    def __init__(self):
        r"""
            This complex relu is a generalization of the standard real valued Leaky Relu
            for complex numbers

            \f[
                g(z) = \frac{1}{2} \left( 1 + \cos( \mathrm{arg}(z) ) \right) \cdot z
            \f]
        """
        super(complexRelu, self).__init__()

    def forward(self,z):
        return 0.5*(1+torch.cos(torch.angle(z)))*z
    
class zLogz(torch.nn.Module):
    def __init__(self):
        r"""
            This activation function returns z*log(z) for all complex numbers, except for the negative real axis
        """
        super(zLogz, self).__init__()
    
    def forward(self, z):
       
        a = z
        condition = torch.logical_and(z.imag == 0, z.real <= 0)
        a[condition] = 0
        
        not_condition = torch.logical_not(condition)
        logarithm = a[not_condition].log()
        a[not_condition] = torch.mul(a[not_condition], logarithm)
        return a
        
class fractions(torch.nn.Module):
    def __init__(self):
        r"""
            it is what it is...
        """
        super(fractions, self).__init__()

    def forward(self,z):
        return 1/(1 + (-z).real.exp()) + 1j/(1 + (-z).imag.exp())
    
class generalisedSoftsign(torch.nn.Module):
    def __init__(self):
        r"""
            it is what it is...
        """
        super(generalisedSoftsign, self).__init__()

    def forward(self,z):
        return z/(2 + z.abs())


