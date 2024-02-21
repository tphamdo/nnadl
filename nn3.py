import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch import optim
from torch.utils.data import Dataset

import load_mnist

def vectorized_result(j):
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e

class MNIST_dataset(Dataset):
  def __init__(self, data):
#    tr_x = np.array([np.reshape(x,(28,28)) for x in train_data[0]])
#    tr_y = np.array([vectorized_result(y) for y in train_data[1]])
    self.X = data[0]
    self.Y = data[1]
    print(self.X.shape[0])
    print("loaded", self.X.shape, self.Y.shape)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.Y[idx])

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1,20, kernel_size=5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(20,40, kernel_size=5)
    self.fc1 = nn.Linear(40*4*4, 100)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1,40*4*4)
    x = F.relu(self.fc1(x))
    x = F.log_softmax(self.fc2(x), dim=1)

    return x

def test(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        output = model.forward(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, reduction='sum')

        # get num correctly classified
        pred = output.max(dim=1).indices
        correct += pred.eq(target).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def reshape(data, is_test=False):
  X = np.array([np.reshape(x,(28,28)) for x in data[0]])
  Y = data[1] if is_test else np.array([vectorized_result(y) for y in data[1]])
  return (X,Y)

if __name__ == "__main__":
  args = {
    'batch_size': 10
    'test_batch_size': 10
    'epochs': 60
    'eta': 0.03 #learning rate
    'cuda': False
    'lambda': 0.1 #unused regularization param
  }

  '''
  train_data, validation_data, test_data = load_mnist.load_data()
  # reshape train/test data
  train_data2 = reshape(train_data)
  test_data2 = reshape(test_data, is_test=True)

  mnist_train_data = MNIST_dataset(train_data2)
  mnist_test_data = MNIST_dataset(test_data2)
  training_loader=torch.utils.data.DataLoader(mnist_train_data, batch_size=args['batch_size'], shuffle=True)
  test_loader=torch.utils.data.DataLoader(mnist_test_data, batch_size=args['test_batch_size'], shuffle=True)
  '''
  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('./data', train=True, download=True,
                  transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor() ,
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['test_batch_size'], shuffle=True)

  model = Net()
  criterion = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=args['eta'])

  if args['cuda']:
    model.cuda()


  test(test_loader, model)
  print(test_loader.dataset.shape)
  for epoch in range(args['epochs']):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
      if (args['cuda']):
        data, target = data.cuda(), target.cuda()

      optimizer.zero_grad()

      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[Epoch: {epoch + 1}, Examples: {(batch_idx+1) * args["batch_size"]}] loss: {running_loss / 2000:.4f}')
        running_loss = 0.0

    test(test_loader, model)
