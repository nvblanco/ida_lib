import torch

device = 'cuda'
cuda = torch.device('cuda')
data_types_2d = {"image", "mask", "segmap", "heatmap"}
internal_type = torch.float32
identity = torch.eye(3, 3, device=cuda)
one_torch = torch.ones(1).to(device)
ones_torch = torch.ones(1, 2, device=cuda)
