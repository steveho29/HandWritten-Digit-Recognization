import torch, torchvision, numpy
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image, ImageFilter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.MNIST('MNIST/', download=False, train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = datasets.MNIST('MNIST/', download=False, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=0.0001,
                      momentum=0.9)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_set) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_set),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_set)))
            torch.save(network.state_dict(), 'convomodel.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_set)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_set),
        100. * correct / len(test_set)))


# test()
# for epoch in range(1, n_epochs + 1):
#     train(epoch)
#     test()

def accuracy():
    batch_size = 64
    file = 'convomodel.pth'
    # file = 'model.pt'
    model = Net()
    network_state_dict = torch.load(file)
    model.load_state_dict(network_state_dict)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST('MNIST/', download=False, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    correct_count, all_count = 0, 0
    for images, labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))

accuracy()