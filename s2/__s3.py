from common import *

random.seed()

datas = [ G() for x in range(10)]

xx , yy = train_data_numpy_array(48, 32, datas)

# print(datas[0])
# print(xx[0])
# print(yy[0])

for i in datas:
  print("%s -> %s" % i)