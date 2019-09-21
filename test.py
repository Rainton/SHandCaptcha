import torch

t = torch.tensor([[0, 1, 2], [3, 4, 5], [7, 8, 9]])
print(t)
print(t.size())
print(t.dim())
print(t[2][1])
t[2][1] = 3
print(t)
