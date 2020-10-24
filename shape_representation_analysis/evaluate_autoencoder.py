import math

with open(r"D:\projects\shape\shape_representation_analysis\ConvAE_training\original_input29", "r") as f:
    origin = f.readlines()

with open(r"D:\projects\shape\shape_representation_analysis\ConvAE_training\reconstruction29", "r") as f:
    reconstruction = f.readlines()

error = 0

for i in range(len(origin)):
    error += (float(origin[i]) - float(reconstruction[i])) ** 2
print(math.sqrt(error)/64)
