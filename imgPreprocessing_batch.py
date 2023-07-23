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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def imgDenoise(file_Name, file_Path):
    
    file = os.path.join(file_Path, file_Name)
    # Load the noisy image
    img = cv2.imread(file)

    # Apply denoising using a bilateral filter
    img_Denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    # evaluate the effect of image reinforcement
    # Calculate the MSE between the images
    mse = mean_squared_error(img, img_Denoised)

    # Calculate the PSNR between the images
    psnr = peak_signal_noise_ratio(img, img_Denoised)

    # Calculate the SSIM between the images
    ssim = structural_similarity(img, img_Denoised, channel_axis=-1)
        
    # store the metrics
    evaluation = [mse, psnr, ssim]

    return file_Name, img_Denoised, evaluation


def imgReinforce(file_Name, file_Path):

    file = os.path.join(file_Path, file_Name)  
    # Load the pre-processed image
    img = cv2.imread(file)

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
 
    return file_Name, unsharp_mask, evaluation


def imgResize(file_Name, file_Path):
    
    file = os.path.join(file_Path, file_Name)  
    # Load the pre-processed image
    img = cv2.imread(file)
        
    # Resize frame to desired size (e.g. 224x224)
    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
       
    return file_Name, resized_img


def imgPreProcessing(file_Path, num_Workers):
    start_time = time.time() # record the start time
    total_Loop = len(os.listdir(file_Path))  # Total number of images

    File_List = os.listdir(file_Path)
    print('number of images need to be pre-processed:{}'.format(total_Loop))

    with ThreadPoolExecutor(max_workers=num_Workers) as executor:
        
        # denoise
        print('image denoise start...')
        futures_Denoise    = {executor.submit(imgDenoise, file, file_Path): file for file in File_List}
        num_Loop           = 0  # Counter to keep track of processed images
        evaluation_metrics = pd.DataFrame({'MSE': [], 'PSNR': [], 'SSIM': []})  # store the evaluation metrics
        
        for future in as_completed(futures_Denoise):
            start_time_loop = time.time()
            file = futures_Denoise[future]         
            try:
                file_Name, img_Denoised, evaluation = future.result()
                evaluation_metrics.loc[len(evaluation_metrics)] = evaluation
                # save denoised image(overwrite the input image)
                cv2.imwrite(os.path.join(file_Path, file_Name), img_Denoised)
                
                # calculate time consumption
                num_Loop += 1
                time_spent_loop = time.time() - start_time_loop
                utils.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 1000)

            except Exception as e:
                print(f'Error processing {file} for denoising: {e}')
                
        # Return the evaluation summary of image denoising, show 4 decimal places
        printEvaluationSummary(evaluationSummary(evaluation_metrics))
        
        
        # reinforce
        print('image reinforce start...')
        futures_Reinforce = {executor.submit(imgReinforce, file, file_Path): file for file in File_List}
        num_Loop           = 0  # Counter to keep track of processed images
        evaluation_metrics = pd.DataFrame({'MSE': [], 'PSNR': [], 'SSIM': []})  # store the evaluation metrics
        
        for future in as_completed(futures_Reinforce):
            start_time_loop = time.time()
            file = futures_Reinforce[future]
            try:
                file_Name, img_Reinforced, evaluation = future.result()
                evaluation_metrics.loc[len(evaluation_metrics)] = evaluation
                # save denoised image(overwrite the input image)
                cv2.imwrite(os.path.join(file_Path, file_Name), img_Reinforced)
                
                # calculate time consumption
                num_Loop += 1
                time_spent_loop = time.time() - start_time_loop
                utils.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 1000)

            except Exception as e:
                print(f'Error processing {file} for reinforcement: {e}')
        # Return the evaluation summary of image denoising, show 4 decimal places
        printEvaluationSummary(evaluationSummary(evaluation_metrics))
        
        
        # resize
        print('image resize start...')
        futures_Resize    = {executor.submit(imgResize, file, file_Path): file for file in File_List}
        num_Loop           = 0  # Counter to keep track of processed images
        
        for future in as_completed(futures_Resize):
            start_time_loop = time.time()
            file = futures_Resize[future]
            try:
                file_Name, img_Resized = future.result()
                # save denoised image(overwrite the input image)
                cv2.imwrite(os.path.join(file_Path, file_Name), img_Resized)
                
                # calculate time consumption
                num_Loop += 1
                time_spent_loop = time.time() - start_time_loop
                utils.estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 1000)

            except Exception as e:
                print(f'Error processing {file} for resize: {e}')

    print('data pre-processing finished, time spent: {}s'.format(round(time.time() - start_time, 4)))
    return True
    

# predict dataset pre-processing
def imgPreProcessing_pred(filePath):
    
    video_List = os.listdir(filePath)
    
    for video in video_List:
        frame_Path = os.path.join(filePath, video)
        frameFile_List = os.listdir(frame_Path)
        # 4-grid video
        if len(frameFile_List) == 4:
            path0 = os.path.join(frame_Path, frameFile_List[0])
            path1 = os.path.join(frame_Path, frameFile_List[1])
            path2 = os.path.join(frame_Path, frameFile_List[2])
            path3 = os.path.join(frame_Path, frameFile_List[3])
            imgPreProcessing(path0)
            imgPreProcessing(path1) 
            imgPreProcessing(path2) 
            imgPreProcessing(path3) 
        # 1-grid video
        else:
            imgPreProcessing(frame_Path)
    
    return True


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
            # save 5 seconds' frame images
            save = time
            for i in range(5):
                frameImg_tmp = video_name + '_' + str(save) + '.jpg'
                frameImgs.append(frameImg_tmp)
                save += save_interval
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
            # save 5 seconds' frame images
            save = time
            for i in range(5):
                frameImg_tmp = video_name + '_' + body_part + '_' + str(save) + '.jpg'
                frameImgs.append(frameImg_tmp)
                save += save_interval
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
    

    
class MyDataSet(Dataset):
    """custom data set"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB is color image, L is grayscale image
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
 
    


    
###############################################################################
### main ###

# def main(args):
    
#     # read the inputs
#     mode           = args.mode
#     frameFile_Path = args.frameFile_Path
#     labelFile_Path = args.labelFile_Path
    
#     if mode == "prediction":
#         imgPreProcessing_pred(frameFile_Path)
#     elif mode == "training":
#         imgPreProcessing(frameFile_Path)
#         classifyImg(labelFile_Path, frameFile_Path, save_interval=1)
#     else:
#         print('mode must be "prediction" or "training"')


# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()

#     # The directory of the test dataset
#     parser.add_argument('--frameFile_Path',  type=str, default="./testData")
#     # The directory of the test label files
#     parser.add_argument('--labelFile_Path',  type=str, default="./testData/file")
#     # mode prediction or training
#     parser.add_argument('--mode',  type=str, default="prediction")

#     opt = parser.parse_args()

#     main(opt)

# for local PC execution
def main(mode='training', frameFile_Path='./', labelFile_Path=''):
    
    # get number of workers
    num_Workers = os.cpu_count() -2
    print(f"Number of available workers: {num_Workers}")
    
    if mode == "prediction":
        imgPreProcessing_pred(frameFile_Path)
    elif mode == "training":
        imgPreProcessing(file_Path=frameFile_Path, num_Workers=num_Workers)
        #classifyImg(labelFile_Path, frameFile_Path, save_interval=1)
    else:
        print('mode must be "prediction" or "training"')
    
    

frameFile_Path = 'C:\\Piloerection\\data\\frameImage\\batch3'
labelFile_Path = args.labelFile_Path

main(mode='training', frameFile_Path=frameFile_Path)

 
    
    
    
    
    
    
    
    
    
    