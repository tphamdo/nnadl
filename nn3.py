import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from torchvision import transforms
from torch import optim
from torch.utils.data import Dataset
#import pickle
#import gzip
import load_mnist

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class MNISTDataset(Dataset):
	def __init__(self, transform=None):
		tr_d, v_d, test_d = load_mnist.load_data()
		tr_x = np.array([np.reshape(x,(1,28,28)) for x in tr_d[0]])
		print(tr_x.shape)
		tr_y = np.array([vectorized_result(y) for y in tr_d[1]])
		print(tr_y.shape)
		self.X = tr_x
		self.Y = tr_y
		self.transform = transform
		print("loaded", self.X.shape, self.Y.shape)

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		sample = (self.X[idx], self.Y[idx])
#		if self.transform:
#			sample = self.transform(sample)

		return sample

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv = nn.Conv2d(1,20, kernel_size=5)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(20*12*12*42, 100)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		print(x.shape)
		x = F.relu(self.conv(x))
		print(x.shape)
		x = self.pool(x)
		print(x.shape)
		x = torch.flatten(x)
		print(x.shape)
		x = F.relu(self.fc1(x))
		print(x.shape)
		x = F.log_softmax(self.fc2(x))

		return x

if __name__ == "__main__":
	device = "cuda"
	eta = 0.1
	epochs = 60

	net = Net()
	print(net.conv.weight.shape)
	mnist_dataset = MNISTDataset()
	trainloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=42, shuffle=True)
	#testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

	criterion = nn.NLLLoss()
	optimizer = optim.SGD(net.parameters(), lr=eta)

	if device == "cuda":
		net.cuda()

	net.train()

	for epoch in range(epochs):

		running_loss = 0.0
		for batch_idx, (data,target) in enumerate(trainloader):
			# get the inputs; data is a list of [inputs, labels]
			#target = target.unsqueeze(-1)
			data, target = data.to(device), target.to(device)
			data = data.float()
			target = target.float()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			output = net(data)
			print(output.shape, target.shape)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
				running_loss = 0.0
