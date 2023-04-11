# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 18:40:41 2023

@author: Chunhui TU
"""

import cv2
import os
import time
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import pandas as pd
import otherTools

def evaluationSummary(df):
    
    # MSE
    max_MSE    = round(df['MSE'].max(),4)
    min_MSE    = round(df['MSE'].min(),4)
    mean_MSE   = round(df['MSE'].mean(),4)
    median_MSE = round(df['MSE'].median(),4)
    
    # PSNR
    max_PSNR    = round(df['PSNR'].max(),4)
    min_PSNR    = round(df['PSNR'].min(),4)
    mean_PSNR   = round(df['PSNR'].mean(),4)
    median_PSNR = round(df['PSNR'].median(),4) 
    
    # SSIM
    max_SSIM    = round(df['SSIM'].max(),4)
    min_SSIM    = round(df['SSIM'].min(),4)
    mean_SSIM   = round(df['SSIM'].mean(),4)
    median_SSIM = round(df['SSIM'].median(),4)
    
    return [[max_MSE, min_MSE, mean_MSE, median_MSE],
            [max_PSNR, min_PSNR, mean_PSNR, median_PSNR],
            [max_SSIM, min_SSIM, mean_SSIM, median_SSIM]]


def printEvaluationSummary(evaluationList):
    
    print('Max MSE:{}; Min MSE:{}; Mean MSE:{}; Median MSE:{}'.format(evaluationList[0][0], 
                                                                      evaluationList[0][1], 
                                                                      evaluationList[0][2], 
                                                                      evaluationList[0][3]))
    
    print('Max PSNR:{}; Min PSNR:{}; Mean PSNR:{}; Median PSNR:{}'.format(evaluationList[1][0], 
                                                                          evaluationList[1][1], 
                                                                          evaluationList[1][2], 
                                                                          evaluationList[1][3]))
    
    print('Max SSIM:{}; Min SSIM:{}; Mean SSIM:{}; Median SSIM:{}'.format(evaluationList[2][0], 
                                                                          evaluationList[2][1], 
                                                                          evaluationList[2][2], 
                                                                          evaluationList[2][3]))
        

def bilateralFilter(file_List, filePath, new_directory):
    
    print('image denoising start...')
    start_time = time.time() # record the start time
    evaluation_metrics = pd.DataFrame({'MSE': [], 'PSNR': [], 'SSIM': []})  # store the evaluation metrics
    num_Loop = 0  # record how many loop was processed
    total_Loop = len(file_List)
    
    for file in file_List:
        
        # record the time spent of 1 loop
        start_time_loop = time.time()
        
        file_name = os.path.join(filePath, file)  
        # Load the noisy image
        img = cv2.imread(file_name)

        # Apply denoising using a bilateral filter
        img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Evaluate the effect of denoising by 
        # evaluate the effect of image reinforcement
        # Calculate the MSE between the images
        mse = mean_squared_error(img, img_denoised)

        # Calculate the PSNR between the images
        psnr = peak_signal_noise_ratio(img, img_denoised)

        # Calculate the SSIM between the images
        ssim = structural_similarity(img, img_denoised, channel_axis=-1)
        
        # store the metrics
        evaluation = [mse, psnr, ssim]
        evaluation_metrics.loc[len(evaluation_metrics)] = evaluation
        
        # save denoised image
        filename = '{}/pre-processed_{}'.format(new_directory, file)
        cv2.imwrite(filename, img_denoised)
        
        # estimate time comsumption
        end_time_loop = time.time()
        time_spent_loop = end_time_loop - start_time_loop
        num_Loop += 1
        otherTools.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 300)
        
    
    # Return the evaluation summary of image denoising, show 4 decimal places
    evaluation_summary = evaluationSummary(evaluation_metrics)

    print('Image Denoising Effect Evaluation')
    printEvaluationSummary(evaluation_summary)
    end_time   = time.time()
    time_spent = end_time - start_time
    print('data denoising finished, time spent: {}s'.format(round(time_spent,4)))    
    return


def imgReinforcement(file_List, filePath):
    
    print('image reinforcement start...')
    start_time = time.time() # record the start time
    evaluation_metrics = pd.DataFrame({'MSE': [], 'PSNR': [], 'SSIM': []})  # store the evaluation metrics
    num_Loop = 0  # record how many loop was processed
    total_Loop = len(file_List)
    
    for file in file_List:
        
        # record the time spent of 1 loop
        start_time_loop = time.time()

        file_name = os.path.join(filePath, file)  
        # Load the pre-processed image
        img = cv2.imread(file_name)

        # Apply the unsharp mask filter to the image
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        unsharp_mask = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
        
        # evaluate the effect of image reinforcement
        # Calculate the MSE between the images
        mse = mean_squared_error(img, unsharp_mask)

        # Calculate the PSNR between the images
        psnr = peak_signal_noise_ratio(img, unsharp_mask)

        # Calculate the SSIM between the images
        ssim = structural_similarity(img, unsharp_mask, channel_axis=-1)
        
        # store the metrics
        evaluation = [mse, psnr, ssim]
        evaluation_metrics.loc[len(evaluation_metrics)] = evaluation

        # Save the processed image(overwrite the input image)
        cv2.imwrite(file_name, unsharp_mask)
        
        # estimate time comsumption
        end_time_loop = time.time()
        time_spent_loop = end_time_loop - start_time_loop
        num_Loop += 1
        otherTools.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 300)

    # Return the evaluation summary of image denoising, show 4 decimal places
    evaluation_summary = evaluationSummary(evaluation_metrics)

    print('Image Reinforcement Evaluation')
    printEvaluationSummary(evaluation_summary)
    end_time   = time.time()
    time_spent = end_time - start_time
    print('data reinforcement finished, time spent: {}s'.format(round(time_spent,4)))    
    return

def RGB2GRAYnResizeImg(file_List, filePath):
    
    print('image resize start...')
    start_time = time.time() # record the start time
    num_Loop = 0  # record how many loop was processed
    total_Loop = len(file_List)
    
    for file in file_List:
        
        # record the time spent of 1 loop
        start_time_loop = time.time()

        file_name = os.path.join(filePath, file)  
        # Load the pre-processed image
        img = cv2.imread(file_name)
        
        # Convert the RGB image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize frame to desired size (e.g. 224x224)
        resized_img = cv2.resize(gray_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Save the processed image(overwrite the input image)
        cv2.imwrite(file_name, resized_img)
        
        # estimate time comsumption
        end_time_loop = time.time()
        time_spent_loop = end_time_loop - start_time_loop
        num_Loop += 1
        otherTools.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 300)
        
    end_time   = time.time()
    time_spent = end_time - start_time
    print('data resize finished, time spent: {}s'.format(round(time_spent,4)))    
    return
    
    
### Main Function ###
def imgPreProcessing(filePath):
    
    start_time = time.time() # record the start time
    
    # create a new sub-directory in current directory to store the pre-processed pictures
    new_directory = os.path.join(filePath,'pre-processed')
    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
        print(f"{new_directory} created successfully!")
    
    originalImg_Path  = os.path.join(filePath,'original')
    originalFile_List = os.listdir(originalImg_Path)
    print('number of images need to be pre-precessed:{}'.format(len(originalFile_List)))
    
    # denoising
    bilateralFilter(originalFile_List, originalImg_Path, new_directory)
    
    # new file list after denoising
    file_List = os.listdir(new_directory)
    
    # reinforcement
    imgReinforcement(file_List, new_directory)
    
    # convert to grayscale and resize image
    RGB2GRAYnResizeImg(file_List, new_directory)
    
    # time spent
    end_time   = time.time()
    time_spent = end_time - start_time
    print('datd pre-processeing finished, pre-processed images saved in {}, time spent: {}s'.format(new_directory, round(time_spent,4)))
    
    return True
    
    
filePath = 'D:\\downloads\\Piloerection\\video\\1-grid_videos\\video2img' 
imgPreProcessing(filePath)   
    

file_List = os.listdir(filePath)
# RGB2GRAYnResizeImg(file_List, filePath)
    

    
    
    
    
    
    