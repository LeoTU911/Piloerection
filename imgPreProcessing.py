# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:40:41 2023

@author: Chunhui TU
"""

import cv2
import os
import time
import statistics



def bilateralFilter(filePath, new_directory):
    
    print('image denoising start...')
    start_time = time.time() # record the start time
    file_List = os.listdir(filePath)
    psnr_List = []
    
    for file in file_List:
        
        file_path = os.path.join(filePath, file)  
        # Load the noisy image
        img = cv2.imread(file_path)

        # Apply denoising using a bilateral filter
        img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Evaluate the effect of denoising by 
        # Compute PSNR between original and denoised images
        psnr = cv2.PSNR(img, img_denoised)
        psnr_List.append(psnr)
        
        # save denoised image
        filename = '{}/denoised_{}.jpg'.format(new_directory, file)
        cv2.imwrite(filename, img_denoised)
    
    # Return the statistical results of PSNR, display 4 decimal places
    max_psnr    = round(max(psnr_List),4)
    min_psnr    = round(min(psnr_List),4)
    mean_psnr   = round(statistics.mean(psnr_List),4)
    median_psnr = round(statistics.median(psnr_List),4)

    print('Image Denoising Effect Evaluation')
    print('Max PSNR:{}; Min PSNR:{}; Mean PSNR:{}; Median PSNR:{}'.format(max_psnr, min_psnr, mean_psnr, median_psnr))
    end_time   = time.time()
    time_spent = end_time - start_time
    print('data denoising finished, time spent: {}s'.format(round(time_spent,4)))    
    return


def imgPreProcessing(filePath):
    
    start_time = time.time() # record the start time
    
    # create a new sub-directory in current directory to store the denoised pictures
    new_directory = os.path.join(filePath,'pre-processed')
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
        print(f"{new_directory} created successfully!")
    
    originalImg_Path = os.path.join(filePath,'original')
    file_List = os.listdir(originalImg_Path)
    print('number of images need to be pre-precessed:{}'.format(len(file_List)))
    
    # denoising
    bilateralFilter(originalImg_Path, new_directory)
    
    # time spent
    end_time   = time.time()
    time_spent = end_time - start_time
    print('datd pre-processeing finished, pre-processed images saved in {}, time spent: {}s'.format(new_directory, round(time_spent,4)))
    
    return True
    
    
filePath = 'D:\\downloads\\Piloerection\\video\\1-grid_videos\\video2img' 
imgPreProcessing(filePath)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    