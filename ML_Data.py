import torch
import h5py as h5


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, fn, param):
        super(TrainDataset, self).__init__()

        Nt = param['nt']

        if param['lattice'] == "two_sites":
            Nx = 2
        else:
            raise RuntimeError( f"TrainDataset not implemented for lattice: {param['lattice']}" )

        with h5.File(fn) as h5f:
            self.raw_action = torch.from_numpy(
                h5f[f"TrainData/TP_action"][()],
            )
            self.raw_flowed_action = torch.from_numpy(
                h5f[f"TrainData/flow_{param['tau_f']:.2e}/TP_action"][()],
            )

            Nconf = len(self.raw_action)
            self.raw = torch.from_numpy(
                h5f[f"TrainData/TP_conf"][()],
            ).reshape(Nconf,Nt,Nx)
            self.raw_flowed = torch.from_numpy(
                h5f[f"TrainData/flow_{param['tau_f']:.2e}/TP_conf"][()],
            ).reshape(Nconf,Nt,Nx)

        self.size = Nconf

    def to(self,*args,**kwargs):
        self.raw = self.raw.to(*args,**kwargs)
        self.raw_action = self.raw_action.to(*args,**kwargs)
        self.raw_flowed = self.raw_flowed.to(*args,**kwargs)
        self.raw_flowed_action = self.raw_flowed_action.to(*args,**kwargs)

        return self

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.raw[item,:], self.raw_flowed[item,:]

    def getAction(self,item):
        return self.raw_action[item], self.raw_flowed_action[item]
