import torch 

# the code is absolutely not optimized for GPU nor could the 
# 2 site problem fill a GPU thus it might be best to keep it 
# with device = 'cpu'.
torchTensorArgs = {
    "device": torch.device('cpu'),
    "dtype" : torch.cdouble
}
