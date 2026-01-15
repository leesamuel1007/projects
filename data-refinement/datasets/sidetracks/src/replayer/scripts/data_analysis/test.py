import numpy as np


a = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [7,8,9],
              [4,5,6]])
print(np.diff(a.sum(axis=-1)))
print(np.ediff1d(a.sum(axis=-1)))
print(np.diff(a, axis=0))