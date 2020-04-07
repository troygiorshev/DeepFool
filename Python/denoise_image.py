import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
#import imageio
import os
import glob
from PIL import Image
from PIL import ImageFilter
import progressbar
import keras

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
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return blur


def medianblur(filepath):
    img = cv2.imread(filepath)
    blur = cv2.medianBlur(img, 5)
    return blur


def bilateralfilter(filepath):
    img = cv2.imread(filepath)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


def MNISTdae(filepath, dae):
    img = cv2.imread(filepath)
    # plt.imshow(img)
    # plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[2:-2, 2:-2] / 255.0
    img = img.reshape(-1, 28, 28, 1)
    #plt.imshow(np.squeeze(img), cmap=cm.gray)
    #plt.show()
    encodedImg = dae.predict(img)
    encodedImg = np.squeeze(encodedImg)
    #plt.imshow(encodedImg, cmap=cm.gray)
    #plt.show()
    return encodedImg*255.0


def adjust_images(input_path, output_folder, dae=None):
    files = [f for f in glob.glob(input_path + "**/*.jpeg", recursive=True)]

    bar = progressbar.ProgressBar(maxval=len(files)).start()
    i = 0
    for path in files:
        filename = os.path.basename(path)

        """
        #IMAGENET Manipulations
        img = denoiseColor(path) 
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'/denoised/'
        cv2.imwrite(savefile+filename, img)

        img = sharpen(path)
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'/sharpen/'
        img.save(savefile+filename)

        img = bilateralfilter(path) 
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'/bilateralfilter/'
        cv2.imwrite(savefile+filename, img)

        img = gaussianblur(path)
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'/gaussianblur/'
        cv2.imwrite(savefile+filename, img)

        img = medianblur(path)
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'/medianblur/'
        cv2.imwrite(savefile+filename, img)
        """

        #MNIST Manipulations

        img = MNISTdae(path, dae)
        savefile = '../data/MNIST/'+output_folder+'/dae/'
        cv2.imwrite(savefile+filename, img)

        img = denoiseColor(path) 
        savefile='../data/MNIST/'+output_folder+'/denoised/'
        cv2.imwrite(savefile+filename, img)


        img = bilateralfilter(path) 
        savefile='../data/MNIST/'+output_folder+'/bilateralfilter/'
        cv2.imwrite(savefile+filename, img)

        img = gaussianblur(path)
        savefile='../data/MNIST/'+output_folder+'/gaussianblur/'
        cv2.imwrite(savefile+filename, img)

        img = medianblur(path)
        savefile='../data/MNIST/'+output_folder+'/medianblur/'
        cv2.imwrite(savefile+filename, img)

        img = sharpen(path)
        savefile='../data/MNIST/'+output_folder+'/sharpen/'
        img.save(savefile+filename)

        i = i+1
        bar.update(i)


def main():
    """
    #IMAGENET
    # path to perturbed images
    input_path = '../data/perturbed/1/'
    output_folder = 'perturbedModification'
    adjust_images(input_path, output_folder)

    input_path = '../data/ILSVRC2012_img_val/'
    output_folder = 'originalImgModification'
    adjust_images(input_path, output_folder)
    """

    #MNIST
    dae = keras.models.load_model('dae_mnist_autoencoder.h5')

    input_path = '../data/MNIST/perturbed/'
    output_folder = 'perturbedModification'
    adjust_images(input_path, output_folder, dae)


if __name__ == '__main__':
    main()
