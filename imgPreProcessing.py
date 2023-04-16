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
import utils
import numpy as np
import shutil

from PIL import Image
import torch
from torch.utils.data import Dataset

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
        utils.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 300)
        
    
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
        utils.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 300)

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
        
        # # Convert the RGB image to grayscale
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize frame to desired size (e.g. 224x224)
        resized_img = cv2.resize(gray_img, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Save the processed image(overwrite the input image)
        cv2.imwrite(file_name, resized_img)
        
        # estimate time comsumption
        end_time_loop = time.time()
        time_spent_loop = end_time_loop - start_time_loop
        num_Loop += 1
        utils.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 300)
        
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
    
    
# filePath = 'D:\\downloads\\Piloerection\\video\\1-grid_videos\\video2img' 
# imgPreProcessing(filePath)   
    

# file_List = os.listdir(filePath)
# # RGB2GRAYnResizeImg(file_List, filePath)
    
###############################################################################

def preprocessForModel(filePath):

    # get all the frame image names
    file_List = os.listdir(filePath)
    print('number of images need to be precessed:{}'.format(len(file_List)))
    
    images = []
    for file in file_List:
        
        file_name = os.path.join(filePath, file)  
        # Load the pre-processed image
        image = cv2.imread(file_name)
        # Convert to an array of float type
        image = np.float32(image)

        # Normalize pixel values to [0,1]
        result = np.zeros(image.shape, dtype=np.float32)
        cv2.normalize(image, dst = result, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        
        # add normalized image to train dataset
        images.append(result)
        
    return images
    
    
"""
1. 把全部帧图片移动到no目录，创建新的文件夹small和large
2.读csv file，得到分类标准
3.匹配到对应的图片文件
4.根据time，第二列，第三列对应的图片名称找到对应的图片
5.移动图片过去，并删除原图片
"""   

def classifyCriteria(labelFile_Name, labelFile_Path, save_interval=1, grid_video = False): 
    
    # label file name format: e.g. '072_dom calf.csv'
    video_name = labelFile_Name.split('_')[0]
    body_part  = labelFile_Name.split('_')[1].split('.')[0]
    # read 1 csv file, rename column names
    df = pd.read_csv(labelFile_Path, skiprows=[0], names=['time', 'small', 'large'])
        
    # find whether has goosebumps in current file in column 2&3
    small_Criteria = df['small'] == 1
    small_time = list(df.loc[small_Criteria, 'time'])
        
    large_Criteria = df['large'] == 1
    large_time = list(df.loc[large_Criteria, 'time'])
    
    if type(save_interval) == int:
        if len(small_time) != 0:
            for i in range(len(small_time)):
                small_time[i] = int(small_time[i])
        if len(large_time) != 0:
            for j in range(len(large_time)):
                large_time[j] = int(large_time[j])
    
    timeLine = small_time + large_time # goosebumps time pionts
        
    # screen corresponding frame image and move them to a new directory
    # The first 1 represents the start of goosebumps, 
    # the second 1 represents the end of goosebumps, 
    # the third represents the start again, and so on
    # Different labels are updated, and the same label is set to 0
    
    small_images = []
    large_images = []
    isGoosebumps = 0 # default state, 0 means no; 1 means small; 2 means large
    
    # Judging the current classification
    if len(timeLine) == 0: # no goosebumps
        # return empty list directly
        return small_images, large_images
    else:
        for i in range(len(timeLine)):
            time = timeLine[i]
            if i == len(timeLine) -1:
                time_next = time
            else:
                time_next = timeLine[i+1]
                
            if time in small_time:
                if isGoosebumps != 1: # current label is 0 or 2, updated
                    isGoosebumps = 1                   
                    small_images_tmp = stitchImgFileNameList(time, time_next, video_name, body_part, save_interval, grid_video)
                    small_images += small_images_tmp
                else:
                    isGoosebumps = 0  # current and new time point both 1, means small goosebumps ends
            else:
                if isGoosebumps != 2: # current label is 0 or 1, updated
                    isGoosebumps = 2
                    large_images_tmp = stitchImgFileNameList(time, time_next, video_name, body_part, save_interval, grid_video)
                    large_images += large_images_tmp
                else:
                    isGoosebumps = 0  # current and new time point both 2, means small goosebumps ends
    
    return small_images, large_images
                    

            
def stitchImgFileNameList(time, time_next, video_name, body_part, save_interval=1, grid_video = False):
    
    frameImgs = []
    if not grid_video:
        # happens for even time points, only save the frame picture at the current time point
        if time == time_next:
            frameImg_tmp = video_name + '_' + str(time) + '.jpg'
            frameImgs.append(frameImg_tmp)
            return frameImgs
        else:
            while time < time_next:
                # Stitched frame picture file name
                frameImg_tmp = video_name + '_' + str(time) + '.jpg'
                frameImgs.append(frameImg_tmp)
                if type(save_interval) == int:
                    time = int(time + save_interval)
                else:
                    time += save_interval
    else:
        if time == time_next:
            frameImg_tmp = video_name + '_' + body_part + '_' + str(time) + '.jpg'
            frameImgs.append(frameImg_tmp)
            return frameImgs
        else:
            while time < time_next:
                # Stitched frame picture file name
                frameImg_tmp = video_name + '_' + body_part + '_' + str(time) + '.jpg'
                frameImgs.append(frameImg_tmp)
                if type(save_interval) == int:
                    time = int(time + save_interval)
                else:
                    time += save_interval
     
    return frameImgs
        
    
def classifyImg(labelFile_Path, frameFile_Path, save_interval=1):
     
    # create new sub-folders 'no', 'small' and 'large'    
    default_Path = os.path.join(frameFile_Path, "no")
    small_Path   = os.path.join(frameFile_Path, "small")
    large_Path   = os.path.join(frameFile_Path, "large")
    if not os.path.exists(default_Path):
            os.mkdir(default_Path)
            print(f"{default_Path} created successfully!")
    if not os.path.exists(small_Path):
            os.mkdir(small_Path)
            print(f"{small_Path} created successfully!")
    if not os.path.exists(large_Path):
            os.mkdir(large_Path)
            print(f"{large_Path} created successfully!")
            
    # get all the frame file name list
    frameFile_List = os.listdir(frameFile_Path)

    # Move all JPG files to new subdirectory and delete original files
    for fileName in frameFile_List:
        frameImgPath = os.path.join(frameFile_Path, fileName)
        if os.path.isfile(frameImgPath) and fileName.endswith(".jpg") and not os.path.isdir(frameImgPath):
            shutil.move(frameImgPath, os.path.join(frameFile_Path, default_Path, fileName))
    
    # get all the label file
    oneGridlabelFile_Path = os.path.join(labelFile_Path, '1-grid')
    fourGridlabelFile_Path = os.path.join(labelFile_Path, '4-grid')
    oneGridlabelFile_List  = os.listdir(oneGridlabelFile_Path)
    fourGridlabelFile_List = os.listdir(fourGridlabelFile_Path)
    
    # Get the classification standard of each label file
    # 1-grid
    for labelFile in oneGridlabelFile_List:
        labelFile_Path_tmp = os.path.join(oneGridlabelFile_Path, labelFile)
        small_images, large_images = classifyCriteria(labelFile_Name = labelFile, 
                                                      labelFile_Path = labelFile_Path_tmp,
                                                      save_interval  = save_interval,
                                                      grid_video = False)

        if len(small_images) != 0:
            # move images to corresponding directory
            for fileName in small_images:
                # original filepath
                frameImgPath = os.path.join(default_Path, fileName)
                # to new filepath
                if os.path.exists(frameImgPath):
                    shutil.move(frameImgPath, os.path.join(small_Path, fileName))
                
        if len(large_images)!= 0:  
            for fileName in large_images:
                # original filepath
                frameImgPath = os.path.join(default_Path, fileName)
                # to new filepath
                if os.path.exists(frameImgPath):
                    shutil.move(frameImgPath, os.path.join(large_Path, fileName))
            
    # 4-grid
    for labelFile in fourGridlabelFile_List:
        labelFile_Path_tmp = os.path.join(fourGridlabelFile_Path, labelFile)
        small_images, large_images = classifyCriteria(labelFile_Name = labelFile, 
                                                      labelFile_Path = labelFile_Path_tmp,
                                                      save_interval  = save_interval,
                                                      grid_video = True)

        if len(small_images) != 0:
            # move images to corresponding directory
            for fileName in small_images:
                # original filepath
                frameImgPath = os.path.join(default_Path, fileName)
                # to new filepath
                if os.path.exists(frameImgPath):
                    shutil.move(frameImgPath, os.path.join(small_Path, fileName))
                
        if len(large_images)!= 0:    
            for fileName in large_images:
                # original filepath
                frameImgPath = os.path.join(default_Path, fileName)
                # to new filepath
                if os.path.exists(frameImgPath):
                    shutil.move(frameImgPath, os.path.join(large_Path, fileName))
    
    return True


###############################################################################
# labelFile_Path = 'D:\\downloads\\Piloerection\\file' 
# frameFile_Path = 'D:\\downloads\\Piloerection\\frameImage'
# classifyImg(labelFile_Path, frameFile_Path, save_interval=1)
    
    
    
    
# test = [1.0, 2]   
    
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    