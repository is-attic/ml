from common import *

random.seed(1234)

datas = [ G_ccc_ddd() for x in range(10)]

xx , yy = train_data_numpy_array(10, 16, datas)

print(datas[0])
print(xx[0])
print(yy[0])