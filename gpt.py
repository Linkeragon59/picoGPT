import torch
import torch.nn as nn

torch.manual_seed(1337)

B,T,C=4,8,2
x = torch.randn(B,T,C)
x.shape
torch.Size([4, 8, 2])
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev, 0)
print(x[0])
print(xbow[0])
xbow2 = torch.zeros((B,T,C))
mean_mat = torch.tril(torch.ones(T,T))
mean_mat = mean_mat / torch.sum(mean_mat, 1, keepdim=True)
xbow2 = mean_mat @ x
print(xbow2[0])

print(mean_mat)

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0, float('-inf'))

print(wei)

print(nn.functional.softmax(wei, 0))
print(nn.functional.softmax(wei, 1))
print(nn.functional.softmax(wei, -1))