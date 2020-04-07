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
import glob
import csv
import progressbar


def classify_images():
    net = models.googlenet(pretrained=True)
    net.eval()
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    paths = {'../data/ILSVRC2012_img_val/raw/': '../data/ILSVRC2012_img_val/classification/imagenet_classify.csv',
    '../data/ILSVRC2012_img_val/originalImgModification/denoised/':'../data/ILSVRC2012_img_val/classification/imagenet_classify_denoised.csv',
    '../data/ILSVRC2012_img_val/originalImgModification/sharpen/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_sharpened.csv',
    '../data/ILSVRC2012_img_val/originalImgModification/bilateralfilter/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_bilateralfilter.csv',
    '../data/ILSVRC2012_img_val/originalImgModification/gaussianblur/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_gaussianblur.csv',
    '../data/ILSVRC2012_img_val/originalImgModification/medianblur/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_medianblur.csv',
    '../data/ILSVRC2012_img_val/perturbed/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_perturbed.csv', #you may have to change this filepath
    '../data/ILSVRC2012_img_val/perturbedModification/denoised/':'../data/ILSVRC2012_img_val/classification/imagenet_classify_perturbed_denoised.csv',
    '../data/ILSVRC2012_img_val/perturbedModification/sharpen/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_perturbed_sharpened.csv',
    '../data/ILSVRC2012_img_val/perturbedModification/bilateralfilter/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_perturbed_bilateralfilter.csv',
    '../data/ILSVRC2012_img_val/perturbedModification/gaussianblur/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_perturbed_gaussianblur.csv',
    '../data/ILSVRC2012_img_val/perturbedModification/medianblur/': '../data/ILSVRC2012_img_val/classification/imagenet_classify_perturbed_medianblur.csv'
    }

    for input, output in paths.items():
        files = [f for f in glob.glob(input + "**/*.JPEG", recursive=True)]
        bar = progressbar.ProgressBar(maxval=len(files)).start()
        classify = {}
        i = 0
        for path in files:
            img = Image.open(path)
            if (img.mode != "RGB"):
                img = img.convert(mode="RGB")

            image = transform(img)

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


def main():
    classify_images()

if __name__=='__main__':
    main()
        
