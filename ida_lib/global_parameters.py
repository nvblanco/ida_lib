import torch

device = 'cuda'
device = device
if not torch.cuda.is_available() and device is None:
    device = -1
cuda = torch.device('cuda')
data_types_2d = {"image", "mask", "segmap", "heatmap"}
internal_type = torch.float32
identity = torch.eye(3, 3, device=cuda)
one_torch = torch.ones(1).to(device)
ones_torch = torch.ones(1, 2, device=cuda)

ones_2_torch = torch.ones(2, device=cuda)
