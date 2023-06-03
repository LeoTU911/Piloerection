# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 01:15:55 2023

Function Description:
Create a frame image save path
Read all the video files under 1-grid_videos and 4-grid_videos in the specified directory, 
and then save the frame pictures in the specified saving directory 
according to the specified time interval

Example: 
Specify the path: D:\downloads\Piloerection\video
Save path: D:\downloads\Piloerection\frameImage

@author: Chunhui TU
"""

import cv2
import os
import time
import argparse

# 1-Grid video to image
def video2Img1Grid(filePath, save_path, save_interval=1, video_list=None):
    """
    Function Description:
        
    Save the picture of the frame of the input video according to the specified time interval
    default save time interval is 1 second
    """     
    if video_list == None:
        # get all the video names
        predict_mode = False
        video_list = os.listdir(filePath)
    else:
        predict_mode = True
        
    print('Videos to be converted:{}'.format(video_list))
    
    for video in video_list:
        video_path = os.path.join(filePath, video)   
        # accept the input video file
        video_capture = cv2.VideoCapture(video_path)
        video_name = video.split('.')[0]
        
        # print the current processing video's name
        print(video)
        # get the video's frame info
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        print('frame rate is：', frame_rate)

        # get the resolution
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('resolution is：', frame_width, 'x', frame_height)
    
        # Image save interval, save according to the time interval specified by the input
        num_Threshold = frame_rate*save_interval
        frame_number  = 0
        i             = 0 # Count, store the picture after the condition is met
        save_number   = 0 # record the number of saved pictures
    
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
    
            # Convert the frame to a picture
            picture = frame
        
            if i%num_Threshold==0:
                # Save the picture to a file
                # predict mode
                if predict_mode:
                    filename = '{}/{}.jpg'.format(save_path, save_number*save_interval)
                # train mode
                else:
                    filename = '{}/{}_{}.jpg'.format(save_path, video_name, save_number*save_interval)
                cv2.imwrite(filename, picture)
                i = 1
                save_number += 1
            else:  
                i += 1
            frame_number += 1   
    
        print('{} pictures saved in {}'.format(save_number-1, save_path))
        
    # release resource
    video_capture.release()
    return True


def GridImg2Imgs(save_path, video, save_number, save_interval, GridImg, save_path_list=None):

    # Crop the 4-grid picture into 4 separate pictures using the defined dimensions
    pic1 = GridImg[0:540, 0:720]       # upper left corner
    pic2 = GridImg[0:540, 750:1470]    # upper right corner
    pic3 = GridImg[550:1080, 0:720]    # lower left corner
    pic4 = GridImg[550:1080, 750:1470] # lower right corner
    
    # predict mode
    if save_path_list != None:
        # Save the 4 separate pictures
        # r thigh
        filename1 = '{}/{}.jpg'.format(save_path_list[0], save_number*save_interval)
        cv2.imwrite(filename1, pic1)
        # dom arm
        filename2 = '{}/{}.jpg'.format(save_path_list[2], save_number*save_interval)
        cv2.imwrite(filename2, pic2)
        # l thigh
        filename3 = '{}/{}.jpg'.format(save_path_list[1], save_number*save_interval)
        cv2.imwrite(filename3, pic3)
        # dom calf
        filename4 = '{}/{}.jpg'.format(save_path_list[3], save_number*save_interval)
        cv2.imwrite(filename4, pic4)
        
    # train mode
    else:
        # Save the 4 separate pictures
        filename1 = '{}/{}_r thigh_{}.jpg'.format(save_path, video, save_number*save_interval)
        cv2.imwrite(filename1, pic1)
        filename2 = '{}/{}_dom arm_{}.jpg'.format(save_path, video, save_number*save_interval)
        cv2.imwrite(filename2, pic2)
        filename3 = '{}/{}_l thigh_{}.jpg'.format(save_path, video, save_number*save_interval)
        cv2.imwrite(filename3, pic3)
        filename4 = '{}/{}_dom calf_{}.jpg'.format(save_path, video, save_number*save_interval)
        cv2.imwrite(filename4, pic4)
    
    return

# 4-Grid video to image
def video2Img4Grid(filePath, save_path, save_interval=1, video_list=None, save_path_list=None):
    """
    Function Description:
        
    Save the picture of the frame of the input video according to the specified time interval
    Crop the 4-grid picture into 4 separate pictures using the defined dimensions
    default save time interval is 1 second
    """  
    
    if video_list == None:
        # get all the video names
        predict_mode = False
        video_list = os.listdir(filePath)
    else:
        predict_mode = True
        
    print('Videos to be converted:{}'.format(video_list))
    
    for video in video_list:
        video_path = os.path.join(filePath, video)   
        # accept the input video file
        video_capture = cv2.VideoCapture(video_path)
        video_name = video.split('.')[0]
        
        # print the current processing video's name
        print(video)
        # get the video's frame info
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        print('frame rate is：', frame_rate)

        # get the resolution
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('resolution is：', frame_width, 'x', frame_height)
    
        # Image save interval, save according to the time interval specified by the input
        num_Threshold = frame_rate*save_interval
        frame_number  = 0
        i             = 0 # Count, store the picture after the condition is met
        save_number   = 0 # record the number of saved pictures
    
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
    
            # Convert the frame to a picture
            picture = frame
        
            if i%num_Threshold==0:
                # Save the picture to a file
                # predict mode
                if predict_mode:
                    GridImg2Imgs(save_path, video_name, save_number, save_interval, picture, save_path_list)
                # train mode
                else:
                    GridImg2Imgs(save_path, video_name, save_number, save_interval, picture)
                i = 1
                save_number += 1
            else:
                i += 1
            frame_number += 1   
    
        print('{} pictures saved in {}'.format((save_number-1)*4, save_path))
        
    # release resource
    video_capture.release()
    return True

### Main Function ###
def video2Img(filePath):
    
    start_time = time.time()
    
    oneGridVideoPath  = os.path.join(filePath,'1-grid_videos')
    fourGridVideoPath = os.path.join(filePath,'4-grid_videos')
    # frame Image save path
    parent_dir = os.path.dirname(filePath)
    frameImgSavePath = os.path.join(parent_dir, "frameImage")
    if not os.path.exists(frameImgSavePath):
            os.mkdir(frameImgSavePath)
            print(f"{frameImgSavePath} created successfully!")
    
    # 1-grid video process
    video2Img1Grid(filePath = oneGridVideoPath, save_path = frameImgSavePath, save_interval=1)
    # 4-Grid video to image
    video2Img4Grid(filePath = fourGridVideoPath, save_path = frameImgSavePath, save_interval=1)
    
    end_time = time.time()
    time_spent = end_time - start_time
    print('video to frame image finished, images saved in {}, time spent: {}s'.format(frameImgSavePath, round(time_spent,4)))
    
    return True


def video2Img_Pred(filePath):
    """
    The function is the same as that of the training set, 
    but the directory storage structure is different
    
    e.g. test video: 024.wmv, 070.wmv
    directory storage structure:
        - testData
            - file
                - 1-grid
                    024_.csv
                - 4-grid
                    070_XX.csv
                    070_XX.csv                   
            - frameImage
                - 024
                    1.jpg
                - 070
                    - dom arm
                        1.jpg
                    - dom calf
                        1.jpg
                    - l thigh
                        1.jpg
                    - r thigh
                        1.jpg
            - video
                - 1-grid_videos
                    024.wmv
                - 4-grid_videos
                    070.wmv
    """
    start_time = time.time()
    
    oneGridVideoPath  = os.path.join(filePath,'1-grid_videos')
    fourGridVideoPath = os.path.join(filePath,'4-grid_videos')
    oneGridVideoList  = os.listdir(oneGridVideoPath)
    fourGridVideoList = os.listdir(fourGridVideoPath)
    # frame Image save path
    parent_dir = os.path.dirname(filePath)
    frameImgSavePath = os.path.join(parent_dir, "frameImage")
    if not os.path.exists(frameImgSavePath):
            os.mkdir(frameImgSavePath)
    
    # 1-grid video process(one by one)
    if len(oneGridVideoList) > 0:
        for oneGridVideo in oneGridVideoList:
            oneGridVideoFrameImgSavePath = os.path.join(frameImgSavePath, oneGridVideo.split('.')[0])
            if not os.path.exists(oneGridVideoFrameImgSavePath):
                os.mkdir(oneGridVideoFrameImgSavePath)
            video2Img1Grid(filePath = oneGridVideoPath, save_path = oneGridVideoFrameImgSavePath, save_interval=1, video_list=[oneGridVideo])

    # 4-Grid video to image(one by one)
    if len(fourGridVideoList) > 0:
        for fourGridVideo in fourGridVideoList:
            fourGridVideoFrameImgSavePath = os.path.join(frameImgSavePath, fourGridVideo.split('.')[0])
            if not os.path.exists(fourGridVideoFrameImgSavePath):
                    os.mkdir(fourGridVideoFrameImgSavePath)
            # 4 sub dir
            fourGridrThighSavePath = os.path.join(fourGridVideoFrameImgSavePath, 'r thigh')
            if not os.path.exists(fourGridrThighSavePath):
                    os.mkdir(fourGridrThighSavePath)
            fourGridlThighSavePath = os.path.join(fourGridVideoFrameImgSavePath, 'l thigh')
            if not os.path.exists(fourGridlThighSavePath):
                    os.mkdir(fourGridlThighSavePath)
            fourGridDomArmSavePath = os.path.join(fourGridVideoFrameImgSavePath, 'dom arm')
            if not os.path.exists(fourGridDomArmSavePath):
                    os.mkdir(fourGridDomArmSavePath)
            fourGridDomCalfSavePath = os.path.join(fourGridVideoFrameImgSavePath, 'dom calf')
            if not os.path.exists(fourGridDomCalfSavePath):
                    os.mkdir(fourGridDomCalfSavePath)
            
            save_path_list = [fourGridrThighSavePath, fourGridlThighSavePath, fourGridDomArmSavePath, fourGridDomCalfSavePath]
            
            video2Img4Grid(filePath = fourGridVideoPath, 
                           save_path = fourGridVideoFrameImgSavePath, 
                           save_interval=1, 
                           video_list=[fourGridVideo],
                           save_path_list = save_path_list)
    
    end_time = time.time()
    time_spent = end_time - start_time
    print('video to frame image finished, images saved in {}, time spent: {}s'.format(frameImgSavePath, round(time_spent,4)))
    
    return True
    

###############################################################################

def main(args):
    
    # read the inputs
    mode           = args.mode
    file_Path = args.file_Path
    
    if mode == "prediction":
        video2Img_Pred(file_Path)
    elif mode == "training":
        video2Img(file_Path)
    else:
        print('mode must be "prediction" or "training"')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # The directory of the test dataset
    parser.add_argument('--file_Path',  type=str, default="./testData")
    # mode prediction or training
    parser.add_argument('--mode',  type=str, default="prediction")

    opt = parser.parse_args()

    main(opt)

