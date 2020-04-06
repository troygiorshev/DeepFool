import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tensorflow as tflow
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
from collections import OrderedDict


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self):
        super(F5, self).__init__()

        """
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))
        """
        # Modified to remove softmax, as required by DeepFool
        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, 10))
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2() 
        self.c2_2 = C2() 
        self.c3 = C3() 
        self.f4 = F4() 
        self.f5 = F5() 

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output

    
# Load data
batch_size_test = 1000

random_seed = 1
torch.manual_seed(random_seed)  # Deterministic

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=False,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.Resize(32),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
        batch_size=batch_size_test, shuffle=True)

# Network you're using (can change to whatever)
net = LeNet5()
net.load_state_dict(torch.load("../models/MNIST/LeNet/model.pth"))

# Switch to evaluation mode
net.eval()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

k = 1

for batch_idx, (data, target) in enumerate(test_loader):
    for im, label in zip(data, target):

        clip = lambda x: clip_tensor(x, 0, 255)

        tf = transforms.Compose([transforms.Normalize(mean = (0,),
                                                      std = ((1/0.3081),)),
                                 transforms.Normalize(mean = (-.1307,), 
                                                      std=(1,)),
                                 transforms.Lambda(clip),
                                 transforms.ToPILImage(),
                                 transforms.Resize(32)])

        # Save original image in "../data/MNIST/orig" (MNIST/raw already contains the weird .gz things)
        if (os.path.exists('../data/MNIST/orig') != 1):
            os.mkdir('../data/MNIST/orig')
        tf(im).save(
                    '../data/MNIST/orig/' + str(k) + '.JPEG')

        """
        print(im.size())
        print(im)
        plt.figure()
        plt.imshow(tf(im))
        plt.title("original")
        plt.show()
        """

        r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

        print("Original label = ", label_orig)
        print("Perturbed label = ", label_pert)

        """
        print(pert_image[0].size())
        print(pert_image[0])
        plt.figure()
        plt.imshow(tf(pert_image.cpu()[0]))
        plt.title(label_pert)
        plt.show()
        """

        # Write image file to directory to hold perturbed images
        if (os.path.exists('../data/MNIST/perturbed') != 1):
            os.mkdir('../data/MNIST/perturbed')
        tf(pert_image.cpu()[0]).save(
                    '../data/MNIST/perturbed/' + str(k) + '.JPEG')
        k += 1
