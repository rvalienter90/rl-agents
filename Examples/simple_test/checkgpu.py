import torch
print(torch.__version__)
print(torch.cuda.is_available())
#print('Device:', torch.device('cuda:0'))
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
