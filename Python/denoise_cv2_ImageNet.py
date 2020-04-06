import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import os
import glob
import multiprocessing

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
# Multiprocessing
num_proc = 6

base = "../data/ILSVRC2012_img_val/"
#in_path = base + "perturbed/"
#save_path = base + "denoised_cv2_perturbed/"  # MAKE THIS FOLDER BEFORE RUNNING
in_path = base + "raw/"
save_path = base + "denoised_cv2_raw/"  # MAKE THIS FOLDER BEFORE RUNNING

def denoiseColor(filepath):
    img = cv2.imread(filepath) 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
    return dst

def do_work(files):
    for path in files:
        img = denoiseColor(path)
        cv2.imwrite(save_path+os.path.basename(path), img)

# Multiprocessing doesn't work without this `__name__ == "__main__"` guard!
# (On Windows, apparently)
if __name__ == "__main__":
    files = [f for f in glob.glob(in_path + "**/*.jpeg", recursive=True)]
    num_files = len(files)
    print(num_files)   # Sanity check

    proc_list = []
    size = num_files // num_proc

    """
    # Check to make sure indexing is right
    for i in range(num_proc):
        if i != num_proc - 1:
            print(i*size, (i+1)*size)
        else:
            print(i*size, "end")
    """

    for i in range(num_proc):
        if i != num_proc - 1:
            files_part = files[i*size:(i+1)*size]
        else:
            files_part = files[i*size:]
        p = multiprocessing.Process(target=do_work, args=(files_part, ))
        proc_list.append(p)

    for p in proc_list:
        p.start()

    # Wait for them all to be done before exiting
    for p in proc_list:
        p.join()

    print("Done")