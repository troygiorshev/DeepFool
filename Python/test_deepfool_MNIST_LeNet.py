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
from deepfool_MNIST import deepfool
import os


class Net(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 6,
                               kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels = 6,
                               out_channels = 16,
                               kernel_size = 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
# Load data
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)  # Deterministic

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(32),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
        batch_size=batch_size_test, shuffle=True)

# Network you're using (can change to whatever)
net = Net(10)
net.load_state_dict(torch.load("../models/MNIST/LeNet/model.pth"))

# Switch to evaluation mode
net.eval()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

for batch_idx, (data, target) in enumerate(test_loader):

    for im, label in zip(data, target):
        #im = im.float()
        r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

        print("Original label = ", label_orig)
        print("Perturbed label = ", label_pert)
            
        clip = lambda x: clip_tensor(x, 0, 255)

        tf =  transforms.Compose([transforms.Normalize(mean = (0,),
                                                       std = ((1/0.3081),)), transforms.Normalize(mean = (-.1307,), std=(1,)),
                    transforms.Lambda(clip),
            transforms.ToPILImage(),
                                  transforms.Resize(32)])
        print(pert_image[0])
        plt.figure()
        plt.imshow(tf(pert_image.cpu()[0]))
        plt.title(label_pert)
        plt.show()

        # Write image file to directory to hold perturbed images
        if (os.path.exists('../data/MNIST/perturbed') != 1):
            os.mkdir('../data/MNIST/perturbed')
        tf(pert_image.cpu()[0]).save(
                    '../data/MNIST/perturbed/' + 'test.', 'JPEG')
        quit()


