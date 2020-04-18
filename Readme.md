# CMPE 351 Spring 2020 - Project

## Group 17

* Chelsey Kurylo
* Jessica Landon
* Troy Giorshev

## Attribution

Cloned from <https://github.com/LTS4/DeepFool>, following [DeepFool](https://arxiv.org/abs/1511.04599) (Full Citation Below).

## Dependencies

Using [conda](https://docs.conda.io/en/latest/miniconda.html), at version 4.8.2 on both Windows 10 and Ubuntu 18.04.

* `conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch -y`
* `conda install matplotlib opencv progressbar2 keras -y`

## Workflow

Note that **all** scripts are in `Python/` and as such we omit `Python/` from the paths here.

### ImageNet LSVRC 2012, Google LeNet

* Download ImageNet LSVRC 2012 Validation Set from [http://www.image-net.org/index](here) or [https://academictorrents.com/collection/imagenet-2012](here).
  * Extract to `data/ILSVRC2012_img_val/raw/` (so the first image is `data/ILSVRC2012_img_val/raw/ILSVRC2012_val_00000001.JPEG`)
* Run `test_deepfool_ImageNet.py` (That's `Python/test_deepfool_ImageNet.py`, as per above)
  * This uses the pretrained Google LeNet model included in torchvision, as per the DeepFool Paper.
  * This populates `data/ILSVRC2012_img_val/perturbed/`
  * This creates `data/ILSVRC2012_img_val/orig_names_and_labels.csv` and `data/ILSVRC2012_img_val/pert_names_and_labels.csv`
  * Note the Average robustness output in `data/ISLVRC2012_img_val/rho.txt`
* Run `denoise_ImageNet.py`
  * This populates the subfolders of `data/ILSVRC2012_img_val/originalImgModification/` and `data/ILSVRC2012_img_val/perturbedModification/`
* Run `classify_ImageNet.py`
  * This populates `data/ILSVRC2012_img_val/classification/`

### MNIST, LeNet

* Run `test_deepfool_MNIST_LeNet.py`
  * We've already found and trained a LeNet model in pytorch.  You can find it in `models/MNIST/LeNet/`.  It was trained with `train_MNIST_LeNet.py`.
  * This populates `data/MNIST_FC/perturbed/`
  * This creates `data/MNIST_LeNet/orig_names_and_labels.csv` and `data/MNIST_LeNet/pert_names_and_labels.csv`
  * Note the Average robustness output in `data/MNIST_LeNet/rho.txt`
* Run `denoise_MNIST_LeNet.py`
  * This populates the subfolders of `data/MNIST_LeNet/originalImgModification/` and `data/MNIST_LeNet/perturbedModification/`
* Run `classify_MNIST_LeNet.py`
  * This populates the `data/MNIST_LeNet/classification/`

### MNIST, Fully Connected 500-150-10

* Run `test_deepfool_MNIST_FC.py`
  * We've already created and trained a LeNet model in pytorch.  You can find it in `models/MNIST/FC/`.  It was trained with `train_MNIST_FC.py`.
  * This populates `data/MNIST_FC/perturbed/`
  * This creates `data/MNIST_FC/orig_names_and_labels.csv` and `data/MNIST_FC/pert_names_and_labels.csv`
  * Note the Average robustness output in `data/MNIST_FC/rho.txt`
* Run `denoise_MNIST.py`
  * This populates the subfolders of `data/MNIST_FC/originalImgModification/` and `data/MNIST_FC/perturbedModification/`
* Run `classify_MNIST_FC.py`
  * This populates `data/MNIST_FC/classification/`

## Reference
[1] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.
