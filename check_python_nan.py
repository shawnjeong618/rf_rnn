import numpy as np
import torch

x = torch.randn(3, 2)

print(x)

cond = x > 0
y = torch.where(cond, x, 0)

print(y)

z = x[cond] 

print(z)

w = torch.zeros_like(x)
w[cond] = z

print(w)