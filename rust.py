import torch

# x = torch.rand(1,2,4,3)
# y = torch.rand(1,2,4,3)
# z = [x,y]
# print(torch.cat(z,1).shape)

s = torch.tensor([[2,3],[4,2],[3,3]])
sc = s * 3
print(sc)
