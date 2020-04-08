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
from deepfool import deepfool
import os
import csv

def main():
    # Number of images to perturb, 0 means "all of them"
    N = 0
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

    j = 0

    # Network you're using (can change to whatever)
    net = models.googlenet(pretrained=True)

    # Switch to evaluation mode
    net.eval()

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv*torch.ones(A.shape))
        A = torch.min(A, maxv*torch.ones(A.shape))
        return A

    base = "../data/ILSVRC2012_img_val/"

    # Get list of files in ImageNet directory (MAKE SURE `base` ABOVE IS CORRECT)

    # First get the list of already perturbed files
    pert_walk_gen = os.walk(base + "perturbed/", topdown = True)
    _, _, pert_files_raw = next(pert_walk_gen)
    pert_files = [p for p in pert_files_raw if p != ".gitignore"] # Remove .gitignore

    # Now process the files
    raw_walk_gen = os.walk(base + "raw/", topdown=True)
    _, _, files_raw = next(raw_walk_gen)
    files_tmp = [f for f in files_raw if f != ".gitignore"] # Remove .gitignore
    files = [f for f in files_tmp if not (f in pert_files)] # Don't process pictures that have already been perturbed

    sorted_files = sorted(files, key=lambda item: int(item[18:23]))

    end = N if N != 0 else len(sorted_files)

    try:
        # Now for the correct number of images
        for name in sorted_files[0:end]:
            orig_img = Image.open(base + "raw/" + name)
            
            # Preprocessing only works for colour images
            if (orig_img.mode != "RBG"):
                orig_img = orig_img.convert(mode="RGB")

            if (orig_img.mode == "RGB"):    # Belt-and-suspenders check
                mean = [ 0.485, 0.456, 0.406 ]
                std = [ 0.229, 0.224, 0.225 ]

                # Get vector form of image so L2 norm of image x can be calculated (See denominator of eqn 15 in DeepFool paper)
                img_arr = np.array(orig_img)
                img_vect = img_arr.ravel()
                L2_norms.append(np.linalg.norm(img_vect))

                # Remove the mean
                im = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean,
                                        std = std)])(orig_img)
                
                r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

                # Add L2 norm of perturbation to array (See numerator of eqn 15 in DeepFool paper)
                r_norm = np.linalg.norm(r)
                r_arr.append(r_norm)

                # Add original labels and perturbed labels to array (just in case you need them later, not rlly using rn)
                orig_labels.append(label_orig)
                pert_labels.append(label_pert)

                labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

                str_label_orig = labels[np.int(label_orig)].split(',')[0]
                str_label_pert = labels[np.int(label_pert)].split(',')[0]

                print("Original label = ", str_label_orig)
                print("Perturbed label = ", str_label_pert)
                print("Number of Iterations = ", loop_i)
            
                clip = lambda x: clip_tensor(x, 0, 255)

                tf =  transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = [(1/0.229), (1/0.244), (1/0.255)]), 
                                    transforms.Normalize(mean = [-0.485, -0.456, -0.406], std=[1,1,1]),
                                    transforms.Lambda(clip), 
                                    transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224)])

                # Add to cumulative sum term to get rho (See eqn 15 in DeepFool paper)
                rho_sum = rho_sum + r_norm / np.linalg.norm(img_vect)

                # Write image file to directory to hold perturbed images
                if (os.path.exists(base + 'perturbed/') != 1):
                    os.mkdir(base + 'perturbed/')
                tf(pert_image.cpu()[0]).save(
                    base + 'perturbed/' + name, '.JPEG')

                # Create .csv with original saved image name and predicted label
                with open(base + 'orig_names_and_labels.csv', 'a') as orig_f:
                    writer = csv.writer(orig_f, lineterminator='\n')
                    writer.writerow([name, str(label_orig)])

                # Create .csv with perturbed saved image name and predicted label
                with open(base + 'pert_names_and_labels.csv', 'a') as pert_f:
                    writer = csv.writer(pert_f, lineterminator='\n')
                    writer.writerow([name, str(label_pert)])

                j += 1
            else:
                raise TypeError(f"expected Image Mode RBG, got {orig_img.mode}")
    except KeyboardInterrupt:
        pass

    # Compute average robustness (rho) for the simulation (See eqn 15 in DeepFool paper)
    rho = (1/j)*rho_sum
    print(f"Number of samples: {j}, Average robustness: {rho}")

if __name__ == "__main__":
    main()
