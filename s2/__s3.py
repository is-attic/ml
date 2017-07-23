from common import *

random.seed()

BATCH_SIZE = 48
datas = [ G() for x in range(BATCH_SIZE)]

xx , yy = train_data_numpy_array(BATCH_SIZE, 32, datas)

# print(datas[0])
# print(xx[0])
# print(yy[0])

for i in datas:
  print("%s -> %s" % i)