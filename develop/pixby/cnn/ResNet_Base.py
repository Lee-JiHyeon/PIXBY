''' 1. Module Import ''' 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


import math
# from tensorboardX import SummaryWriter

# ''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

# # Hyper Parameter
# BATCH_SIZE = 32
# EPOCHS = 35

# # learning rate decay(step) parameter
# STEP_SIZE = 5
# GAMMA = 0.1

# model_weight_save_path = "/content/drive/My Drive/model/"
dataset_path = 'C:\\Users\\multicampus\\Desktop\\SSAKIT\\Project\\back\\stl-10'


''' 3. CIFAR10 데이터 다운로드(Train set, Test set 분리) - Data Augmentation 적용 (1) 기본 cifar-10 데이터 '''
# train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
#                                  train = True,
#                                  download = True,
#                                  transform = transforms.Compose([
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.5, 0.5, 0.5),
#                                      (0.5, 0.5, 0.5))                            
#                                  ]))
# test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
#                                  train = False,
#                                  transform = transforms.Compose([
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.5, 0.5, 0.5),
#                                      (0.5, 0.5, 0.5))                            
#                                  ]))
# train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#                                            batch_size = BATCH_SIZE,
#                                            shuffle = True)
# test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
#                                            batch_size = BATCH_SIZE,
#                                            shuffle = False)


# traindir = os.path.join(dataset_path, 'train')

# data set path에 테스트할 데이터의 경로를 올려야한다. test는 테스트 폴더 이름
# testdir = os.path.join(dataset_path, 'test')

# train_dataset = datasets.ImageFolder(traindir,
#                                     transform = transforms.Compose([
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.5, 0.5, 0.5),
#                                         (0.5, 0.5, 0.5))                            
#                                     ]))
test_dataset = datasets.ImageFolder(dataset_path,
                                    transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5))                            
                                    ]))
# train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#                                            batch_size = BATCH_SIZE,
#                                            shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 32,
                                           shuffle = False)


''' 4. 데이터 확인 (1) '''
# for (X_train, Y_train) in train_loader:
#   print('X_train:', X_train.size(), 'type:', X_train.type())
#   print('Y_train:', Y_train.size(), 'type:', Y_train.type())
#   break
# print(len(train_dataset))
print(len(test_dataset))


''' 5. 데이터 확인 (2) '''
# pltsize = 1
# plt.figure(figsize=(10 * pltsize, pltsize))

# for i in range(10):
#   plt.subplot(1, 10, i + 1)
#   plt.axis('off')
#   plt.imshow(np.transpose(X_train[i], (1, 2, 0)))
#   plt.title('Class: ' + str(Y_train[i].item()))




''' 6. ResNet 모델 설계 '''

# class BasicBlock(nn.Module):
#   def __init__(self, in_planes, planes, stride = 1):
#     super(BasicBlock, self).__init__()
#     self.conv1 = nn.Conv2d(in_planes, planes,
#                            kernel_size = 3,
#                            stride = stride,
#                            padding = 1,
#                            bias = False)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.conv2 = nn.Conv2d(planes, planes,
#                            kernel_size = 3,
#                            stride = 1,
#                            padding = 1,
#                            bias = False)
#     self.bn2 = nn.BatchNorm2d(planes)

#     # shortcut 정의
#     self.shortcut = nn.Sequential()
#     # 차원이 다른경우
#     if stride != 1 or in_planes != planes:
#       self.shortcut = nn.Sequential(
#           nn.Conv2d(in_planes, planes,
#                     kernel_size = 1,
#                     stride = stride,
#                     bias = False),
#           nn.BatchNorm2d(planes)
#       )

#   def forward(self, x):
#     out = F.relu(self.bn1(self.conv1(x)))
#     out = self.bn2(self.conv2(out))
#     # skip connection
#     out += self.shortcut(x)
#     out = F.relu(out)
#     return out



# class ResNet(nn.Module):
#   def __init__(self, num_classes = 10):
#     super(ResNet, self).__init__()
#     self.in_planes = 16

#     self.conv1 = nn.Conv2d(3, 16,
#                            kernel_size = 3,
#                            stride = 1,
#                            padding = 1,
#                            bias = False)
#     self.bn1 = nn.BatchNorm2d(16)
#     self.layer1 = self._make_layer(16, 2, stride = 1)
#     self.layer2 = self._make_layer(32, 2, stride = 2)
#     self.layer3 = self._make_layer(64, 2, stride = 2)
#     self.linear = nn.Linear(64, num_classes)

#   def _make_layer(self, planes, num_blocks, stride):
#     strides = [stride] + [1] * (num_blocks - 1)
#     layers = []
#     for stride in strides:
#       layers.append(BasicBlock(self.in_planes, planes, stride))
#       self.in_planes = planes
#     return nn.Sequential(*layers)

#   def forward(self, x):
#     out = F.relu(self.bn1(self.conv1(x)))
#     out = self.layer1(out)
#     out = self.layer2(out)
#     out = self.layer3(out)
#     out = F.adaptive_avg_pool2d(out, 1)
#     out = out.view(out.size(0), -1)
#     out = self.linear(out)
#     return out

''' 모델 불러오기 '''
model = torch.load('../../ColabNotebooksresnet_base_all (1).tar')
model.eval()


''' 7. Optimizer, Objective Fucntion 설정 '''
# model = ResNet().to(DEVICE)
# SGD(확률적 경사하강법) 적용, L2 Regularization 적용
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Step decay 적용
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
criterion = nn.CrossEntropyLoss()
# lrs = []

print(model)


''' 8. ResNet 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
# def train(model, train_loader, optimizer, log_interval):
#   model.train()
#   for batch_idx, (image, label) in enumerate(train_loader):
#     image = image.to(DEVICE)
#     label = label.to(DEVICE)
#     optimizer.zero_grad()
#     output = model(image)
#     loss = criterion(output, label)
#     loss.backward()
#     optimizer.step()

#     if batch_idx % log_interval == 0:
#       print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
#           Epoch, batch_idx * len(image),
#           len(train_loader.dataset), 100. * batch_idx / len(train_loader),
#           loss.item()))


''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for image, label in test_loader:
      image = image.to(DEVICE)
      label = label.to(DEVICE)
      output = model(image)
      test_loss += criterion(output, label).item()
      prediction = output.max(1, keepdim = True)[1]
      correct += prediction.eq(label.view_as(prediction)).sum().item()
  test_loss /= len(test_loader.dataset)
  test_accuracy = 100. * correct / len(test_loader.dataset)
  return test_loss, test_accuracy

print(evaluate())

''' 10. ResNet 모델 학습을 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
# for Epoch in range(1, EPOCHS + 1):
#   train(model, train_loader, optimizer, log_interval=200)
#   lrs.append(optimizer.param_groups[0]["lr"])
#   scheduler.step()
#   test_loss, test_accuracy = evaluate(model, test_loader)
#   print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".
#         format(Epoch, test_loss, test_accuracy))

# train, loss 그래프 출력
# plt.plot(range(1, EPOCHS + 1), accuracy)

# learning rate 그래프 출력
# plt.plot(range(1, EPOCHS + 1),lrs)

''' 모델 저장하기 '''
# PATH = '/content/drive/My Drive/ColabNotebooks'
# torch.save(model.state_dict(), PATH + 'model_weights.pth')

# writer = SummaryWriter(logdir='scalar/sin&cos')

# for step in range(-360, 360):
#   angle_rad = step * math.pi / 180
#   writer.add_scalar('sin', math.sin(angle_rad), step)
#   writer.add_scalar('cos', math.cos(angle_rad), step)

# writer.close()

# # %load_ext tensorboard
# # %tensorboard --logdir scalar --port=6006

# for step in range(-360, 360):
#   angle_rad = step * math.pi / 180
#   writer.add_scalars('sin and cos', {'sin': math.sin(angle_rad), 'cos': math.cos(angle_rad)}, step)

# writer.close()

# %load_ext tensorboard

# %tensorboard --logdir scalar --port=6006