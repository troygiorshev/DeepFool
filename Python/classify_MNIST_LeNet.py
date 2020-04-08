import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
import os
#import imageio
import glob
import csv
import progressbar
from train_MNIST_LeNet import LeNet5

#torch.set_num_threads(2)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A


def classify_images():
    random_seed = 1
    torch.manual_seed(random_seed)
    net = LeNet5()
    net.load_state_dict(torch.load("../models/MNIST/LeNet/model.pth"))
    net.eval()
    clip = lambda x: clip_tensor(x, 0, 255)
    transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])

    paths = {
    '../data/MNIST_LeNet/orig/': '../data/MNIST_LeNet/classification/MNIST_classify.csv',
    '../data/MNIST_LeNet/originalImgModification/denoised/':'../data/MNIST_LeNet/classification/MNIST_classify_denoised.csv',
    '../data/MNIST_LeNet/originalImgModification/sharpen/': '../data/MNIST_LeNet/classification/MNIST_classify_sharpened.csv',
    '../data/MNIST_LeNet/originalImgModification/bilateralfilter/': '../data/MNIST_LeNet/classification/MNIST_classify_bilateralfilter.csv',
    '../data/MNIST_LeNet/originalImgModification/gaussianblur/': '../data/MNIST_LeNet/classification/MNIST_classify_gaussianblur.csv',
    '../data/MNIST_LeNet/originalImgModification/medianblur/': '../data/MNIST_LeNet/classification/MNIST_classify_medianblur.csv',
    '../data/MNIST_LeNet/originalImgModification/dae/': '../data/MNIST_LeNet/classification/MNIST_classify_dae.csv',
    '../data/MNIST_LeNet/perturbed/': '../data/MNIST_LeNet/classification/MNIST_classify_perturbed.csv',
    '../data/MNIST_LeNet/perturbedModification/denoised/':'../data/MNIST_LeNet/classification/MNIST_classify_perturbed_denoised.csv',
    '../data/MNIST_LeNet/perturbedModification/sharpen/': '../data/MNIST_LeNet/classification/MNIST_classify_perturbed_sharpened.csv',
    '../data/MNIST_LeNet/perturbedModification/bilateralfilter/': '../data/MNIST_LeNet/classification/MNIST_classify_perturbed_bilateralfilter.csv',
    '../data/MNIST_LeNet/perturbedModification/gaussianblur/': '../data/MNIST_LeNet/classification/MNIST_classify_perturbed_gaussianblur.csv',
    '../data/MNIST_LeNet/perturbedModification/medianblur/': '../data/MNIST_LeNet/classification/MNIST_classify_perturbed_medianblur.csv',
    '../data/MNIST_LeNet/perturbedModification/dae/': '../data/MNIST_LeNet/classification/MNIST_classify_perturbed_dae.csv'
    }

    num_classes = 10

    for input, output in paths.items():
        files = [f for f in glob.glob(input + "**/*.jpeg", recursive=True)]
        bar = progressbar.ProgressBar(maxval=len(files)).start()
        classify = {}
        i = 0
        for path in files:
            image = Image.open(path)
            image = transform(image)

            #image = image.view(-1, 28*28)
            f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
            I = (np.array(f_image)).flatten().argsort()[::-1]

            label = I[0]

            classify[os.path.basename(path)]=label
            i = i+1
            bar.update(i)

        with open(output, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for row in classify.items():
                writer.writerow(row)
        print('FINISHED: '+output)


def main():
    classify_images()

if __name__=='__main__':
    main()
