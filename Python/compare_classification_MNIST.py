import os
import imageio
import glob
import csv
import pandas as pd
import numpy as np 
from functools import reduce


def compare():

    path = "../data/MNIST_FC/classification/"
    img = pd.read_csv(path+"MNIST_classify.csv", names=['image_name', 'img']) 
    img_pert = pd.read_csv(path+"MNIST_classify_perturbed.csv", names=['image_name', 'pert']) 
    img_pert_denoise = pd.read_csv(path+"MNIST_classify_perturbed_denoised.csv", names=['image_name', 'pert_denoised']) 
    img_pert_gauss = pd.read_csv(path+"MNIST_classify_perturbed_gaussianblur.csv", names=['image_name', 'pert_gauss']) 
    img_pert_sharp = pd.read_csv(path+"MNIST_classify_perturbed_sharpened.csv", names=['image_name', 'pert_sharp']) 
    img_pert_med = pd.read_csv(path+"MNIST_classify_perturbed_medianblur.csv", names=['image_name', 'pert_med']) 
    img_pert_bf = pd.read_csv(path+"MNIST_classify_perturbed_bilateralfilter.csv", names=['image_name', 'pert_bf']) 
    img_pert_dae = pd.read_csv(path+"MNIST_classify_perturbed_dae.csv", names=['image_name', 'pert_dae']) 
    img_denoise = pd.read_csv(path+"MNIST_classify_denoised.csv", names=['image_name', 'img_denoised']) 
    img_gauss = pd.read_csv(path+"MNIST_classify_gaussianblur.csv", names=['image_name', 'img_gauss']) 
    img_sharp = pd.read_csv(path+"MNIST_classify_sharpened.csv", names=['image_name', 'img_sharp']) 
    img_med = pd.read_csv(path+"MNIST_classify_medianblur.csv", names=['image_name', 'img_med']) 
    img_bf = pd.read_csv(path+"MNIST_classify_bilateralfilter.csv", names=['image_name', 'img_bf'])
    img_dae = pd.read_csv(path+"MNIST_classify_dae.csv", names=['image_name', 'img_dae'])  
    print(img.shape)
    print(img_pert.shape)
    print(img_pert_denoise.shape)
    print(img_pert_gauss.shape)
    print(img_pert_sharp.shape)
    print(img_pert_med.shape)
    print(img_pert_bf.shape)
    print(img_denoise.shape)
    print(img_sharp.shape)
    print(img_med.shape)
    print(img_bf.shape)
    dfs = [img ,img_pert,img_pert_denoise,img_pert_gauss,img_pert_sharp,img_pert_med,
        img_pert_bf , img_pert_dae, img_denoise,img_gauss,img_sharp,img_med,img_bf, img_dae]
    df = reduce(lambda left,right: pd.concat([left,right], axis=1), dfs)
    print(df.head())
    print(df.shape)

    df["img_pert_cp"] = np.where(df["img"] == df["pert"], 1, 0)
    df["img_pert_denoise_cp"] = np.where(df["img"] == df["pert_denoised"], 1, 0)
    df["img_pert_gauss_cp"] = np.where(df["img"] == df["pert_gauss"], 1, 0)
    df["img_pert_sharp_cp"] = np.where(df["img"] == df["pert_sharp"], 1, 0)
    df["img_pert_med_cp"] = np.where(df["img"] == df["pert_med"], 1, 0)
    df["img_pert_bf_cp"] = np.where(df["img"] == df["pert_bf"], 1, 0)
    df["img_pert_dae_cp"] = np.where(df["img"] == df["pert_dae"], 1, 0)
    df["img_denoise_cp"] = np.where(df["img"] == df["img_denoised"], 1, 0)
    df["img_gauss_cp"] = np.where(df["img"] == df["img_gauss"], 1, 0)
    df["img_sharp_cp"] = np.where(df["img"] == df["img_sharp"], 1, 0)
    df["img_med_cp"] = np.where(df["img"] == df["img_med"], 1, 0)
    df["img_bf_cp"] = np.where(df["img"] == df["img_bf"], 1, 0)
    df["img_dae_cp"] = np.where(df["img"] == df["img_dae"], 1, 0)


    df1 = df[['img_pert_cp','img_pert_denoise_cp', 'img_pert_gauss_cp',
        'img_pert_sharp_cp', 'img_pert_med_cp', 'img_pert_bf_cp','img_pert_dae_cp',
        'img_denoise_cp', 'img_gauss_cp','img_sharp_cp', 'img_med_cp','img_bf_cp',
        'img_dae_cp']]
    
    print(df1.sum(axis = 0))

def main():
    compare()

if __name__ == '__main__':
    main()
