import numpy as np

a0 = np.array([[[1, 2], [1, 2], [1, 2]]], dtype=np.int32)
a1 = np.array([[1,2,3],[1,2,3]], dtype=np.int32)

print(a0)
print(a1)

print(np.matmul(a0, a1))

a2 = np.array([[1,2],[1,2],[1,2]], dtype=np.int32)
a3 = np.array([4,5], dtype = np.int32)
print(np.add(a2, a3))

a4 = np.arange(1, 25, dtype=np.int32).reshape([4,2,3])
a5 = np.arange(13, 19, dtype=np.int32).reshape([3,2])
print(np.matmul(a4, a5))