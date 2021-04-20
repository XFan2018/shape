from matplotlib import pyplot as plt
import numpy as np
import torch

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_title("resnet18 vs VGG5self-attention")
# ax.set_ylabel("accuracy")
# ax.set_xlabel("beta")
# ax.set_xticks(np.arange(0, 3, 0.5))
# ax.legend()
# beta = [0.5, 1.0, 1.5, 2.0, 2.5]
# acc_resnet = [65.29, 72.06, 72.64, 69.11, 68.82]
# acc_vgg5self_attention = [70.59, 70.29, 71.76, 72.94, 67.35]
# ax.plot(beta, acc_resnet, '-o', label="resnet")
# ax.plot(beta, acc_vgg5self_attention, '-x', label="VGG5self_attention")
# # annotate
# for i in range(len(acc_resnet)):
#     ax.annotate(acc_resnet[i], (beta[i], acc_resnet[i]))
#     ax.annotate(acc_vgg5self_attention[i], (beta[i], acc_vgg5self_attention[i]))
# plt.grid(axis='y')
# plt.legend()
# plt.savefig("resnet18_vs_VGG5self-attention")
from shape_representation_analysis.neural_network import VGG6PolygonCoordinates, VGG5PolygonCoordinatesSelfAttention, \
    PreActResNet18

VGG6 = VGG6PolygonCoordinates(8, 16, 32, 128, 128, 64)
VGG5Attention = VGG5PolygonCoordinatesSelfAttention(8, 16, 32, 128, 64, 2, 4)
resnet18 = PreActResNet18()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == "__main__":
    print("VGG6:", get_n_params(VGG6))
    print("VGG5Attention:", get_n_params(VGG5Attention))
    print("resnet18:", get_n_params(resnet18))
