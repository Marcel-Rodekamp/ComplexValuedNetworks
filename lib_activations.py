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


