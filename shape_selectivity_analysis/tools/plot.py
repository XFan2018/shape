import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20)
loss = []
top5 = []
adversary_top1 = []
adversary_top5 = []
with open(r"D:\projects\shape\shape_representation_analysis\log_training_ConvAE3_circular_padding_8_16_32\train_epoch_loss.txt", 'r') as f:
    for line in f.readlines():
        line = line.split()
        loss.append(line[1])
        # adversary_top1.append(line[3])
        # adversary_top5.append(line[4])

plt.title("loss vs epochs")
plt.ylabel("loss")
plt.xlabel("epochs")
top1 = [float(i) for i in loss[1:]]
# adversary_top1 = [float(i) for i in adversary_top1[1:]]
# adversary_top5 = [float(i) for i in adversary_top5[1:]]

plt.plot(x, top1, "ko-", label="loss")
# plt.plot(x, adversary_top1, label="adversary top1")
# plt.plot(x, adversary_top5, label="adversary top5")
plt.legend()
plt.show()

print(top1)
# print(adversary_top1)
# print(adversary_top5)
