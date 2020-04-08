# CMPE 351 Spring 2020 - Project

## Group 17

* Chelsey Kurylo
* Jessica Landon
* Troy Giorshev

## Attribution

Cloned from <https://github.com/LTS4/DeepFool>, following [DeepFool](https://arxiv.org/abs/1511.04599).

## Dependencies

Using [conda](https://docs.conda.io/en/latest/miniconda.html), at version 4.8.2 on both Windows 10 and Ubuntu 18.04.

* `conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch -y`
* `conda install matplotlib opencv progressbar2 keras -y`

## Workflow

Note that **all** scripts are in `Python/` and as such we omit `Python/` from the paths here.

### ImageNet LSVRC 2012, Google LeNet

* Download ImageNet LSVRC 2012 Validation Set from [http://www.image-net.org/index](here) or [https://academictorrents.com/collection/imagenet-2012](here).
  * Extract to `data/ILSVRC2012_img_val/raw/` (so the first image is `data/ILSVRC2012_img_val/raw/ILSVRC2012_val_00000001.JPEG`)
* Run `test_deepfool.py` (That's `Python/test_deepfool.py`, as per above)
  * This uses the pretrained Google LeNet model included in torchvision, as per the DeepFool Paper.
  * This populates `data/ILSVRC2012_img_val/perturbed/`
  * Note the Average robustness output
* Run `denoise_ImageNet.py`
  * This populates the subfolders of `data/ILSVRC2012_img_val/originalImgModification/` and `data/ILSVRC2012_img_val/perturbedModification/`
* Run `classify_ImageNet.py`
  * This populates `data/ILSVRC2012_img_val/classification/`

### MNIST, LeNet

* Run `test_deepfool_MNIST_LeNet.py`
  * We've already found and trained a LeNet model in pytorch.  You can find it in `models/MNIST/LeNet/`.  It was trained with `train_MNIST_LeNet.py`.
  * This populates `data/MNIST_FC/perturbed/`
* Run `denoise_MNIST.py`
  * This populates the subfolders of `data/MNIST/originalImgModification/` and `data/MNIST/perturbedModification/`
* Run `classify_MNIST.py`
  * This populates the `data/MNIST/classification/`
