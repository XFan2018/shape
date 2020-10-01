import pickle
import sparse_coding2
import os
import scipy.io as sio

with open('sorted_shapes-120.mat', 'rb') as file:
    shapes = sio.loadmat(file)['shapes']

lambdaT = 0.1
numComp = 120
batchSize = 50

s = sparse_coding2.learn_sparse_components(
    shapes, numComp, lambdaT, batchSize)  # should take a few minutes
print(s)
a = {'component': s.components_}

sio.savemat('basisShapesC' + str(numComp) + 'L' + str(lambdaT), a)

fig1 = sparse_coding2.plot_components(s.components_)
fig1.show()
fig1.savefig('DictionaryC' + str(numComp) + 'L' + str(lambdaT) + '.png', dpi=300)
