import numpy as np

DOMAIN_SIZE = 100

a = np.arange(DOMAIN_SIZE, dtype = np.float32)
b = np.zeros([DOMAIN_SIZE, DOMAIN_SIZE], dtype = np.float32)
a = a.reshape([DOMAIN_SIZE, 1])

for i in range(DOMAIN_SIZE):
  b[i, i] = 1

print(a)