import torch
from lib.models.LMModule import LMModule
torch.set_num_threads(2)
torch.manual_seed(0)
directions=torch.randn(1,64,2)
directions=directions/directions.norm(dim=-1,keepdim=True)
truth=torch.tensor([[0.1,0.2,-0.15]])
c,s=torch.cos(truth[0,0]),torch.sin(truth[0,0])
rotation=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
points=(directions@rotation.T)*(1+torch.rand(1,64,1))+truth[:,None,1:]
points.requires_grad_(True)
solver=LMModule(n_iter=12,device='cpu')
estimate=solver(directions,points)
assert torch.allclose(estimate,truth,atol=1e-3),(estimate,truth)
estimate.square().sum().backward()
assert points.grad is not None and torch.isfinite(points.grad).all()
print('PASS: synthetic 3DoF pose recovery and finite backward gradients')
