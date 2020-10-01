from matplotlib import pyplot as plt
import csv
import numpy as np
import math

x = np.arange(5, 85, 5)
with open("D:\\projects\\summerProject2020\\project2\\finetune_silhouette\\low_resolution_silhouettes.csv", 'r') as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        if idx == 0:
            low_resolution = row
        if idx == 1:
            error1 = row
        if idx == 2:
            original = row
        if idx == 3:
            error2 = row

print(low_resolution)
print(error1)
print(original)
print(error2)
low_resolution = np.array([float(i) for i in low_resolution[1:]])
original = np.array([float(i) for i in original[1:]])
sqrt_n = math.sqrt(20)  # 20 observes
error1 = np.array([float(i) / sqrt_n / 20 for i in error1[1:]])
error2 = np.array([float(i) / sqrt_n / 20 for i in error2[1:]])

#  y += np.random.normal(0, 0.1, size=y.shape)
plt.ylim((0, 1))
plt.title("silhouettes vs 32-sided polygon silhouettes")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.plot(x, low_resolution, label="32-sided")
plt.plot(x, original, 'k-', label="original")
plt.fill_between(x, low_resolution - error1, low_resolution + error1, alpha=0.3)
plt.fill_between(x, original - error2, original + error2, alpha=0.3)
plt.legend()
plt.show()
