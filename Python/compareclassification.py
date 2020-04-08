import os
import imageio
import glob
import csv
import pandas as pd
import numpy as np 
from functools import reduce


def compare():

    path = "../data/ILSVRC2012_img_val/classification/"
    img = pd.read_csv(path+"imagenet_classify.csv", names=['image_name', 'img']) 
    img_pert = pd.read_csv(path+"imagenet_classify_perturbed.csv", names=['image_name', 'pert']) 
    img_pert_denoise = pd.read_csv(path+"imagenet_classify_perturbed_denoised.csv", names=['image_name', 'pert_denoised']) 
    img_pert_gauss = pd.read_csv(path+"imagenet_classify_perturbed_gaussianblur.csv", names=['image_name', 'pert_gauss']) 
    img_pert_sharp = pd.read_csv(path+"imagenet_classify_perturbed_sharpened.csv", names=['image_name', 'pert_sharp']) 
    img_pert_med = pd.read_csv(path+"imagenet_classify_perturbed_medianblur.csv", names=['image_name', 'pert_med']) 
    img_pert_bf = pd.read_csv(path+"imagenet_classify_perturbed_bilateralfilter.csv", names=['image_name', 'pert_bf']) 
    img_denoise = pd.read_csv(path+"imagenet_classify_denoised.csv", names=['image_name', 'img_denoised']) 
    img_gauss = pd.read_csv(path+"imagenet_classify_gaussianblur.csv", names=['image_name', 'img_gauss']) 
    img_sharp = pd.read_csv(path+"imagenet_classify_sharpened.csv", names=['image_name', 'img_sharp']) 
    img_med = pd.read_csv(path+"imagenet_classify_medianblur.csv", names=['image_name', 'img_med']) 
    img_bf = pd.read_csv(path+"imagenet_classify_bilateralfilter.csv", names=['image_name', 'img_bf']) 

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
        img_pert_bf ,img_denoise,img_gauss,img_sharp,img_med,img_bf]
    #df = reduce(lambda left,right: pd.merge(left,right,on=['image_name','image_name'], how='left'), dfs)
    df = reduce(lambda left,right: pd.concat([left,right], axis=1), dfs)
    #df = reduce(lambda left,right: left.merge(right,left_on='image_name', right_on='image_name', how='inner'), dfs)


    print(df.head())
    print(df.shape)

    df["img_pert_cp"] = np.where(df["img"] == df["pert"], 1, 0)
    df["img_pert_denoise_cp"] = np.where(df["img"] == df["pert_denoised"], 1, 0)
    df["img_pert_gauss_cp"] = np.where(df["img"] == df["pert_gauss"], 1, 0)
    df["img_pert_sharp_cp"] = np.where(df["img"] == df["pert_sharp"], 1, 0)
    df["img_pert_med_cp"] = np.where(df["img"] == df["pert_med"], 1, 0)
    df["img_pert_bf_cp"] = np.where(df["img"] == df["pert_bf"], 1, 0)
    df["img_denoise_cp"] = np.where(df["img"] == df["img_denoised"], 1, 0)
    df["img_gauss_cp"] = np.where(df["img"] == df["img_gauss"], 1, 0)
    df["img_sharp_cp"] = np.where(df["img"] == df["img_sharp"], 1, 0)
    df["img_med_cp"] = np.where(df["img"] == df["img_med"], 1, 0)
    df["img_bf_cp"] = np.where(df["img"] == df["img_bf"], 1, 0)


    df1 = df[['img_pert_cp','img_pert_denoise_cp', 'img_pert_gauss_cp','img_pert_sharp_cp', 
    'img_pert_med_cp', 'img_pert_bf_cp','img_denoise_cp', 'img_gauss_cp','img_sharp_cp', 'img_med_cp','img_bf_cp']]
    
    print(df1.sum(axis = 0))


    """
    for i in range(images): 
        c_im = df.iloc[i, 1] 
        c_pert = df.iloc[i, 2]
        c_denoise = df.iloc[i, 3]
        c_gauss = df.iloc[i, 4]
        c_sharp = df.iloc[i, 5]
        c_med = df.iloc[i, 6]
        c_bf = df.iloc[i,7]
        if(c_im == c_pert):
            img_pert += 1
        if(c_im == c_denoise):
            img_den += 1
        if(c_pert == c_denoise):
            pert_den += 1
        if(c_im == c_gauss):
            img_gauss += 1
        if(c_im == c_sharp):
            img_sharp += 1
        if(c_im == c_med):
            img_med += 1
        if(c_im == c_bf):
            img_bf += 1
        #print(df.iloc[i, 0] ,df.iloc[i, 1] ,df.iloc[i, 2], df.iloc[i, 3] )
        #print(images, img_pert, img_den, pert_den)
        #if i>5:
        #    break
    print('images, perturbation, denoise, gaussian, sharpend, median blur, bilateral filter')
    print(images, img_pert, img_den, img_gauss, img_sharp, img_med, img_bf)


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
    """

def main():
    compare()

if __name__ == '__main__':
    main()
