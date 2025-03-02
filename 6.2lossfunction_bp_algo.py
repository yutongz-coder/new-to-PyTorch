import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)


loss1 = L1Loss(reduction='sum')
result1 = loss1(inputs, targets)
loss2 = L1Loss(reduction='mean')
result2 = loss2(inputs, targets)
loss_mse = nn.MSELoss()
result3 = loss_mse(inputs, targets)

print(result1)
print(result2)
print(result3)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
# torch.reshape(x, (1, 3)) 的目的是为了让输入张量 x 的形状符合 nn.CrossEntropyLoss() 损失函数对输入的要求
# 预测值 x：形状为 (N, C)，其中 N 是批量大小（batch size），表示一次处理的样本数量；C 是类别数，即模型预测的类别数量
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)


