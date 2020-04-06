import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import os
import glob
from PIL import Image
from PIL import ImageFilter
import progressbar
import multiprocessing


NUM_PROC = 6


def denoiseColor(filepath):
    img = cv2.imread(filepath) 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
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
    files_all = [f for f in glob.glob(input_path + "**/*.jpeg", recursive=True)]
    num_files = len(files_all)

    proc_list = []
    size = num_files // num_proc

    for i in range(num_proc):
        if i != num_proc - 1:
            files_part = files_all[i*size:(i+1)*size]
        else:
            files_part = files_all[i*size:]
        p = multiprocessing.Process(target=do_work, args=(files_part, output_folder))
        proc_list.append(p)

    for p in proc_list:
        p.start()

    # Wait for them all to be done before exiting
    for p in proc_list:
        p.join()


def do_work(files, output_folder):
    for path in files:
        filename = os.path.basename(path)

        img = denoiseColor(path) 
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'denoised/'
        cv2.imwrite(savefile+os.path.basename(path), img)

        img = sharpen(path)
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'sharpen/'
        img.save(savefile+os.path.basename(path))

        img = bilateralfilter(path) 
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'bilateralfilter/'
        cv2.imwrite(savefile+filename, img)

        img = gaussianblur(path)
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'gaussianblur/'
        cv2.imwrite(savefile+filename, img)

        img = medianblur(path)
        savefile='../data/ILSVRC2012_img_val/'+output_folder+'medianblur/'
        cv2.imwrite(savefile+filename, img) 


def main():
    # path to perturbed images
    input_path = '../data/ILSVRC2012_img_val/perturbed/'
    output_folder = 'perturbedModification/'
    adjust_images(input_path, output_folder)

    print("Finished Perturbed Images")

    input_path = '../data/ILSVRC2012_img_val/raw/'
    output_folder = 'originalImgModification/'
    adjust_images(input_path, output_folder)

    print("Finished All Images")

if __name__=='__main__':
    num_proc = int(input("Number of Threads: "))
    main()

