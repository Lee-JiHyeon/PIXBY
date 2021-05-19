''' 1. Module Import ''' 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import math
# import ResNet_R_X2


class BasicBlock(nn.Module):
  def __init__(self, in_planes, planes, stride = 1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes,
                           kernel_size = 3,
                           stride = stride,
                           padding = 1,
                           bias = False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes,
                           kernel_size = 3,
                           stride = 1,
                           padding = 1,
                           bias = False)
    self.bn2 = nn.BatchNorm2d(planes)

    # shortcut 정의
    self.shortcut = nn.Sequential()
    # 차원이 다른경우
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, planes,
                    kernel_size = 1,
                    stride = stride,
                    bias = False),
          nn.BatchNorm2d(planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    # skip connection
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, num_classes = 10):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64,
                           kernel_size = 3,
                           stride = 1,
                           padding = 1,
                           bias = False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(64, 2, stride = 1)
    self.layer2 = self._make_layer(128, 2, stride = 2)
    self.layer3 = self._make_layer(256, 2, stride = 2)
    self.layer4 = self._make_layer(512, 2, stride = 2)
    self.linear = nn.Linear(512, num_classes)

  def _make_layer(self, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(BasicBlock(self.in_planes, planes, stride))
      self.in_planes = planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.adaptive_avg_pool2d(out, 1)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out



def Infer(dataset_path, model_path):
  folders = os.listdir(dataset_path)
  # ''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
  if torch.cuda.is_available(): 
    DEVICE = torch.device('cuda')
  else:
    DEVICE = torch.device('cpu')

  print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)


  # model_weight_save_path = "/content/drive/My Drive/model/"
  dataset_path = dataset_path

  if dataset_path:
    test_dataset = datasets.ImageFolder(dataset_path,
                                        transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))                            
                                        ]))

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = 32,
                                              shuffle = False)
  else:
    test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                 train = False,
                                 download=True,
                                 transform = transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))                            
                                 ]))
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = 32,
                                           shuffle = False)

  ''' 4. 데이터 확인 (1) 데이터 갯수 확인 '''
  for (X_test, Y_test) in test_loader:
    print('X_test:', X_test.size(), 'type:', X_test.type())
    print('Y_test:', Y_test.size(), 'type:', Y_test.type())
    break

  # print(len(test_dataset))


  ''' 5. 데이터 확인 (2) '''
  # pltsize = 1
  # plt.figure(figsize=(10 * pltsize, pltsize))

  # for i in range(10):
  #   plt.subplot(1, 10, i + 1)
  #   plt.axis('off')
  #   plt.imshow(np.transpose(X_test[i], (1, 2, 0)))
  #   plt.title('Class: ' + str(Y_test[i].item()))



  ''' 모델 불러오기 '''
  # print(type(torch.load(model_path)))
  if type(torch.load(model_path)) == 'dict':
    model = torch.load(model_path)
    # print("check")
  else:
    model = ResNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
  # print(model)
  model.eval()


  ''' 7. Optimizer, Objective Fucntion 설정 '''
  criterion = nn.CrossEntropyLoss()
  # print(model)

  ''' 8-(1). 평가행렬 함수 '''
  test_matrix = [[0] * len(folders) for _ in range(len(folders))]
  def eval_detail(pred, label):
    for i in range(len(label)):
      test_matrix[label[i]][pred[i][0]] += 1


  ''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
  def evaluate(model, test_loader, check=1):
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
        if check:
          eval_detail(prediction, label)
        correct += prediction.eq(label.view_as(prediction)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

  test_loss, test_accuracy = evaluate(model, test_loader)

  # print(test_loss, test_accuracy)
  total = 0
  for test in test_matrix:
    total += sum(test)
  for i in range(len(test_matrix)):
      for j in range(len(test_matrix[i])):
        test_matrix[i][j] = round(test_matrix[i][j]/total,3)
  print(test_matrix)
  return test_loss, test_accuracy, test_matrix 