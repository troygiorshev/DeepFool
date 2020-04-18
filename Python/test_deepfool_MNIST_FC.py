import torch.nn as nn
import torch.nn.functional as F
import torchvision
# import tensorflow as tflow
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
import csv
from itertools import chain

class FC(nn.Module):
    
    def __init__(self):
        super(FC, self).__init__()
        self.l1 = nn.Linear(28*28, 500)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(500, 150)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(150, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        #return F.log_softmax(x)
        return x    # Deepfool needs the model BEFORE softmax
    
# Load data
batch_size_test = 1000

random_seed = 1
torch.manual_seed(random_seed)  # Deterministic

# Unfortunately this grabs the data from "../data/MNIST/processed".
# Whereas everything else is going into "../data/MNIST_FC/"
# This won't be a problem, there will just be a strange lack of raw data in MNIST_FC/

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
        batch_size=batch_size_test, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data/', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
        batch_size=batch_size_test, shuffle=True)

full_loader = chain(train_loader, test_loader)

# Network you're using (can change to whatever)
net = FC()
net.load_state_dict(torch.load("../models/MNIST/FC/model.pth"))

# List to hold L2 norms of r for all perturbed images so rho can be caluclated at the end
r_arr = []
# List to hold original labels
orig_labels = []
# List to hold perturbed labels
pert_labels = []
# List to hold L2 norms
L2_norms = []
# Cumulative sum for rho
rho_sum = 0

# Switch to evaluation mode
net.eval()

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

base = '../data/MNIST_FC/'

k = 1

for batch_idx, (data, target) in enumerate(full_loader):
    for im, label in zip(data, target):

        clip = lambda x: clip_tensor(x, 0, 255)

        tf = transforms.Compose([transforms.Normalize(mean = (0,),
                                                      std = ((1/0.3081),)),
                                 transforms.Normalize(mean = (-.1307,), 
                                                      std=(1,)),
                                 transforms.Lambda(clip),
                                 transforms.ToPILImage()])

        # Save original image in "../data/MNIST_FC/orig" (MNIST/raw already contains the weird .gz things)
        if (os.path.exists(base + 'orig') != 1):
            os.mkdir(base + 'orig')
        tf(im).save(base + 'orig/' + str(k) + '.JPEG')

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
        print("Number of Iterations = ", loop_i)

        """
        print(pert_image[0].size())
        print(pert_image[0])
        plt.figure()
        plt.imshow(tf(pert_image.cpu()[0]))
        plt.title(label_pert)
        plt.show()
        """
        ### RHO CALCULATION
        # Get vector form of image so L2 norm of image x can be calculated (See denominator of eqn 15 in DeepFool paper)
        img_arr = np.array(tf(im))
        img_vect = img_arr.ravel()
        L2_norms.append(np.linalg.norm(img_vect))

        # Add L2 norm of perturbation to array (See numerator of eqn 15 in DeepFool paper)
        r_norm = np.linalg.norm(r)
        r_arr.append(r_norm)

        # Add to cumulative sum term to get rho (See eqn 15 in DeepFool paper)
        rho_sum = rho_sum + r_norm / np.linalg.norm(img_vect)


        # Write image file to directory to hold perturbed images
        if (os.path.exists(base + 'perturbed') != 1):
            os.mkdir(base + 'perturbed')
        tf(pert_image.cpu()[0]).save(base + 'perturbed/' + str(k) + '.JPEG')

        # Create .csv with original saved image name and predicted label
        with open(base + 'orig_names_and_labels.csv', 'a', newline='') as orig_f:
            writer = csv.writer(orig_f)
            writer.writerow([str(k) + '.JPEG', str(label_orig)])

        # Create .csv with perturbed saved image name and predicted label
        with open(base + 'pert_names_and_labels.csv', 'a', newline='') as pert_f:
            writer = csv.writer(pert_f)
            writer.writerow([str(k) + '.JPEG', str(label_pert)])
                
        k += 1


# Compute average robustness (rho) for the simulation (See eqn 15 in DeepFool paper)
rho = (1/(k-1))*rho_sum
print(f"Number of samples: {k-1}, Average robustness: {rho}")
with open(base + "rho.txt", 'a') as rho_f:
    rho_f.write(f"Number of samples: {k}, Average robustness: {rho}\n")
