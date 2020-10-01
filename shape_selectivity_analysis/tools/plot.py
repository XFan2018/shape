import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20, 501, 20)
top1 = []
top5 = []
adversary_top1 = []
adversary_top5 = []
with open("D:\\projects\\summerProject2020\\project3\\log_no_es_test_polygon_1dvgg11_128_128_64\\test_after_train.txt", 'r') as f:
    for line in f.readlines():
        line = line.split()
        top1.append(line[4])
        # adversary_top1.append(line[3])
        # adversary_top5.append(line[4])

plt.ylim((0, 1))
plt.title("VGG11 without early stopping")
plt.ylabel("accuracy")
plt.xlabel("epochs")
top1 = [float(i) for i in top1[:]]
top5 = [float(i) for i in top5[:]]
# adversary_top1 = [float(i) for i in adversary_top1[1:]]
# adversary_top5 = [float(i) for i in adversary_top5[1:]]

plt.plot(x, top1, "ko-", label="top1")
# plt.plot(x, adversary_top1, label="adversary top1")
# plt.plot(x, adversary_top5, label="adversary top5")
plt.legend()
plt.show()

print(top1)
# print(adversary_top1)
# print(adversary_top5)
