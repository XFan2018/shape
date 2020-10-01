import math

with open(r"D:\projects\summerProject2020\project3\ae_training\original_input0") as f:
    origin = f.readlines()

with open(r"D:\projects\summerProject2020\project3\ae_training\reconstruction0") as f:
    reconstruction = f.readlines()

error = 0

for i in range(len(origin)):
    error += (float(origin[i]) - float(reconstruction[i])) ** 2
print(math.sqrt(error)/64)
