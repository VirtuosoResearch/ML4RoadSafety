import numpy as np

test = np.load('METR-LA/train.npz')
# test = np.load('PEMS-BAY/train.npz')
test = np.array(test)
print(test.shape)