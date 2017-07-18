import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


torch.manual_seed(1)

DOMAIN_SIZE = 16

INPUT_SIZE = 1
OUTPUT_SIZE = DOMAIN_SIZE
HIDDEN_SIZE = 16


a = np.arange(DOMAIN_SIZE, dtype = np.float32)
a = a.reshape([DOMAIN_SIZE, 1])

b = np.zeros([DOMAIN_SIZE, DOMAIN_SIZE], dtype = np.float32)
b[:,:] = 0
for i in range(DOMAIN_SIZE):
  b[i, i] = 1

X = torch.from_numpy(a)
Y = torch.from_numpy(b)

x = Variable(X)
y = Variable(Y)

class Net(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
    super(Net, self).__init__()
    self.hidden = torch.nn.Linear(n_feature, n_hidden)
    self.out = torch.nn.Linear(n_hidden, n_output)

  def forward(self, x):
    #x = F.relu(self.hidden(x))
    x = F.sigmoid(self.hidden(x))
    x = self.out(x)
    return x

net = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
loss_func = torch.nn.MSELoss()

target_y = np.arange(DOMAIN_SIZE)

for t in range(30000):
  out = net(x)
  loss = loss_func(out, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


  if t % 1000 == 0:
    prediction = torch.max(F.softmax(out), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    accuracy = sum(pred_y == target_y) / OUTPUT_SIZE
    print(t, accuracy)

out = net(x)
prediction = torch.max(F.softmax(out), 1)[1]
print(prediction)
print(out.data.numpy())
print(np.argmax(out.data.numpy(),1))
#prediction = torch.max(F.softmax(out), 1)

