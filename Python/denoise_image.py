import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import imageio
import os
import glob
from PIL import Image
from PIL import ImageFilter
import progressbar

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


def denoiseGrayscale(filepath):
    img = cv2.imread(filepath) 
    dst = cv2.fastNlMeansDenoising(img, None, 10, 10, 7, 15) 
    return dst


def sharpen(filepath):
    imageObject = Image.open(filepath)
    imageObject = imageObject.filter(ImageFilter.SHARPEN)
    return imageObject


def gaussianblur(filepath):
    img = cv2.imread(filepath)
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur


def medianblur(filepath):
    img = cv2.imread(filepath)
    blur = cv2.medianBlur(img,5)
    return blur


def bilateralfilter(filepath):
    img = cv2.imread(filepath)
    blur = cv2.bilateralFilter(img,9,75,75)
    return blur


def adjust_images(input_path, output_folder):
    files = [f for f in glob.glob(input_path + "**/*.jpeg", recursive=True)]

    bar = progressbar.ProgressBar(maxval=len(files)).start()
    i=1
    for path in files:
        filename = os.path.basename(path)

        img = denoiseColor(path) 
        savefile='../data/'+output_folder+'/denoised/'
        cv2.imwrite(savefile+os.path.basename(path), img)

        img = sharpen(path)
        savefile='../data/'+output_folder+'/sharpen/'
        img.save(savefile+os.path.basename(path))

        img = bilateralfilter(path) 
        savefile='../data/'+output_folder+'/bilateralfilter/'
        cv2.imwrite(savefile+filename, img)

        img = gaussianblur(path)
        savefile='../data/'+output_folder+'/gaussianblur/'
        cv2.imwrite(savefile+filename, img)

        img = medianblur(path)
        savefile='../data/'+output_folder+'/medianblur/'
        cv2.imwrite(savefile+filename, img)
        
        i = i+1
        bar.update(i)


def main():
    # path to perturbed images
    input_path = '../data/perturbed/1/'
    output_folder = 'ILSVRCperturbedModification'
    adjust_images(input_path, output_folder)

    input_path = '../data/ILSVRC2012_img_val/'
    output_folder = 'ILSVRCoriginalImgModification'
    adjust_images(input_path, output_folder)

if __name__=='__main__':
    main()
