#REMOVE ALL MaxPool2d
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim

BATCH_SIZE = 100
CHANNEL = 1
NUM_OF_LABEL = 10
LEARNING_RATE = 0.01
EPOCH = 1

transform = transforms.Compose([transforms.Scale(227),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_set = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class CAM(nn.Module):
  def __init__(self):
    super(CAM,self).__init__()
    self.conv0 = nn.Conv2d(CHANNEL, 96, 11, 4, 0)
    self.relu0 = nn.ReLU()
    #self.pool0 = nn.MaxPool2d(3, 2)
    self.norm0 = nn.BatchNorm2d(96)
    self.conv1 = nn.Conv2d(96, 256, 5, 1, 2)
    self.relu1 = nn.ReLU()
    #self.pool1 = nn.MaxPool2d(3, 2)
    self.norm1 = nn.BatchNorm2d(256)
    self.conv2 = nn.Conv2d(256, 384, 3, 1, 1)
    self.relu2 = nn.ReLU()
    self.conv3 = nn.Conv2d(384, 384, 3, 1, 1)
    self.relu3 = nn.ReLU()
    self.conv4 = nn.Conv2d(384, 256, 3, 1, 1)
    self.relu4 = nn.ReLU()
    self.conv5 = nn.Conv2d(256, 1025, 3, 1, 1)
    self.relu5 = nn.ReLU()
    self.gap = nn.AvgPool2d(13)
    self.fc = nn.Linear(1025, NUM_OF_LABEL)
    self.softmax = nn.Softmax()

  def weight_init(self, mean=0.0, std=0.01):
    for m in self._modules:
      if isinstance(self._modules[m], nn.Conv2d) or isinstance(self._modules[m], nn.Linear):
        self._modules[m].weight.data.normal_(mean, std)
        self._modules[m].bias.data.zero_()

  def forward(self, inputs):
    x = self.conv0(inputs)
    x = self.relu0(x)
    x = self.pool0(x)
    x = self.norm0(x)
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.pool1(x)
    x = self.norm1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.relu3(x)
    x = self.conv4(x)
    x = self.relu4(x)
    x = self.conv5(x)
    x = self.relu5(x)
    x = self.gap(x)
    x = self.fc(x.view(-1,1025))
    output = self.softmax(x)
    return output

alexcam = CAM()
alexcam.weight_init()
alexcam.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexcam.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

for epcoh in range(EPOCH):
  running_loss = 0.0
  for i, data in enumerate(train_loader):
    inputs, labels = data
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    optimizer.zero_grad()
    outputs = alexcam(inputs)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.data[0]
  print running_loss

correct = 0.0
total = 0.0

for data in test_loader:
  images, labels = data
  outputs = alexcam(Variable(images.cuda()))
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %.4f %%' % (
      100.0 * correct / total))

