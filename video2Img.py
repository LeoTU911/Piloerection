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

# 1-Grid video to image
def video2Img1Grid(filePath, save_path, save_interval=1):
    """
    Function Description:
        
    Save the picture of the frame of the input video according to the specified time interval
    default save time interval is 1 second
    """     
    # get all the video names
    video_list = os.listdir(filePath)
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
        save_number   = 1 # record the number of saved pictures
    
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            i += 1
            if not ret:
                break
    
            # Convert the frame to a picture
            picture = frame
        
            if i%num_Threshold==0:
                # Save the picture to a file
                filename = '{}/{}_{}.jpg'.format(save_path, video_name, save_number*save_interval)
                cv2.imwrite(filename, picture)
                i = 0
                save_number += 1
            
            frame_number += 1   
    
        print('{} pictures saved in {}'.format(save_number-1, save_path))
        
    # release resource
    video_capture.release()
    return True


def GridImg2Imgs(save_path, video, save_number, save_interval, GridImg):

    # Crop the 4-grid picture into 4 separate pictures using the defined dimensions
    pic1 = GridImg[0:540, 0:720]       # upper left corner
    pic2 = GridImg[0:540, 750:1470]    # upper right corner
    pic3 = GridImg[550:1080, 0:720]    # lower left corner
    pic4 = GridImg[550:1080, 750:1470] # lower right corner

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
def video2Img4Grid(filePath, save_path, save_interval=1):
    """
    Function Description:
        
    Save the picture of the frame of the input video according to the specified time interval
    Crop the 4-grid picture into 4 separate pictures using the defined dimensions
    default save time interval is 1 second
    """     
    # get all the video names
    video_list = os.listdir(filePath)
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
        save_number   = 1 # record the number of saved pictures
    
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            i += 1
            if not ret:
                break
    
            # Convert the frame to a picture
            picture = frame
        
            if i%num_Threshold==0:
                # Save the picture to a file
                GridImg2Imgs(save_path, video_name, save_number, save_interval, picture)
                i = 0
                save_number += 1
            
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
    


# call the main function
filePath = 'D:\\downloads\\Piloerection\\video'
video2Img(filePath)


