import torch.nn as nn
import torch
import os
torch.manual_seed(os.getenv("SEED"))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# AveCrossEntropyLoss
class FinetuneLoss:
    def __init__(self):
        self.softmax = nn.LogSoftmax(dim=1)

    def __call__(self, inputs):
        inputs = self.softmax(inputs)
        inputs = (-inputs) * (1 / len(inputs[0]))
        return torch.sum(inputs, dim=1).sub(6.90775).sum()


class KLLoss:
    def __init__(self):
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, inputs):
        log_part = self.log_softmax(inputs) * 1000
        p = self.softmax(inputs)
        return torch.sum(log_part * p, dim=1).sum()

#
# x = torch.tensor([[1.,2.,3.,4.], [4.,3.,2.,1.]])
# # print(x)
# log_finetune_1000-500_epoch=30 = nn.LogSoftmax(dim=1)
# log_part = log_finetune_1000-500_epoch=30(x) / 1000
# print(log_part)
# # softmax = nn.Softmax(dim=1)
# # p = softmax(x)
# # print(p)
# result = torch.sum(log_part, dim=1)
# print(result)
# # print("#" * 30)
# loss = FinetuneLoss()
# # # print(loss(x))
# x = torch.ones(1, 1000)
# log_softmax = nn.LogSoftmax()
# print(loss(x))
#
