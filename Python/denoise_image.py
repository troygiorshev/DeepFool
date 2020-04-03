import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
#import imageio
import os
import glob

"""
filepath = "../data/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG"
# Reading image from folder where it is stored 
img = cv2.imread(filepath) 
#img = imageio.imread('data/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG')

# denoising of image saving it into dst image 
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
savefile='../data/denoisedperturbed/'
cv2.imwrite(savefile+os.path.basename(filepath), dst)
# Plotting of source and destination image 
plt.subplot(121), plt.imshow(img) 
plt.subplot(122), plt.imshow(dst) 
  
plt.show()
"""
def denoiseColor(filepath):
    img = cv2.imread(filepath) 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
    return dst

base = "../data/ILSVRC2012_img_val/"
pert_path = base + "perturbed/"
save_path = base + "denoisedperturbed/"

files = [f for f in glob.glob(pert_path + "**/*.jpeg", recursive=True)]

print(len(files))

for path in files:
    img = denoiseColor(path) 
    cv2.imwrite(save_path+os.path.basename(path), img)
