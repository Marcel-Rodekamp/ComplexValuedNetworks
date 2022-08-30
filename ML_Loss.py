import torch

from TwoSite_Hubbard_Model import action, force


class Action(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, expK_p, expK_h, PARAM):
        self.save_for_backward(input_tensor)
        self.expK_p = expK_p
        self.expK_h = expK_h
        self.param = PARAM

        self.N, self.Nt, self.Nx = input_tensor.size()

        return torch.tensor([action(input_tensor[n, :, :], expK_p, expK_h, PARAM) for n in range(self.N)])

    @staticmethod
    def backward(self, grad_input):
        input_tensor, = self.saved_tensors

        forces = torch.zeros((self.N, self.Nt, self.Nx), dtype=torch.cdouble)
        for n in range(self.N):
            forces[n, :, :] = force(input_tensor[n, :, :], self.expK_p, self.expK_h, self.param)

        return grad_input[:, None, None] * forces.conj(), None, None, None


ActionFunc = Action.apply


class SignOptimizedLoss(torch.nn.Module):
    def __init__(self, expK_p, expK_h, param):
        super(SignOptimizedLoss, self).__init__()
        self.PARAM = param
        self.expK_p = expK_p
        self.expK_h = expK_h
        self.a = param['a']
        self.b = param['b']
        self.c = param['c']

        self.DBUG_varImS = []
        self.DBUG_varPhi = []
        self.DBUG_DeltaS = []

    def forward(self, input_tensor, ReS):
        N, _ = input_tensor.size()

        input_tensor = input_tensor.view(N, self.PARAM['Nt'], self.PARAM['Nx'])

        S = ActionFunc(input_tensor, self.expK_p, self.expK_h, self.PARAM)

        # use this to encode d/dt Im S = 0
        var_ImS = self.a*torch.var(S.imag)
        self.DBUG_varImS.append(var_ImS.item())
        var_Phi = self.c/torch.var(S.real).mean()
        self.DBUG_varPhi.append(var_Phi.item())

        # use this to encode d/dt Re S >= 0
        DeltaS = self.b*(S.real-ReS).mean()

        if DeltaS >= 0:
            self.DBUG_DeltaS.append(0)
            return var_ImS + var_Phi
        else:
            self.DBUG_DeltaS.append(DeltaS.item())
            return var_ImS + var_Phi - DeltaS


class HolomorphicLoss(torch.nn.Module):
    def __init__(self, expK_p, expK_h, param):
        super(HolomorphicLoss, self).__init__()
        self.PARAM = param
        self.expK_p = expK_p
        self.expK_h = expK_h

    def forward(self, input_phis, nn_phis, delta_tau):
        N, _ = input_phis.size()

        # finite difference, first order approximation of dPhi/dtau_flow
        LHS = (nn_phis - input_phis)/delta_tau
        RHS = [force(nn_phis[n, :].view(self.PARAM['Nt'], self.PARAM['Nx']),
                   self.expK_p, self.expK_h, self.PARAM).flatten().conj() for n in range(N)]
        RHS = torch.stack(RHS)

        return torch.mean(torch.abs(LHS-RHS))
