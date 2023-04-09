# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 01:15:55 2023

@author: Chunhui TU
"""

import cv2
import os


def video2Img(filePath, save_interval=1):
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
        # create a new sub-directory in current directory to store the frame pictures
        new_directory = os.path.join(filePath,'video2img')
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
            print(f"{new_directory} created successfully!")
        
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
                filename = '{}/{}_{}.jpg'.format(new_directory, video, save_number*save_interval)
                cv2.imwrite(filename, picture)
                i = 0
                save_number += 1
            
            frame_number += 1   
    
        print('{} pictures saved in {}'.format(save_number-1, new_directory))
        
    # release resource
    video_capture.release()
    return True


def GridImg2Imgs(new_directory, video, save_number, save_interval, GridImg):

    # Crop the 4-grid picture into 4 separate pictures using the defined dimensions
    pic1 = GridImg[0:540, 0:720]       # upper left corner
    pic2 = GridImg[0:540, 750:1470]    # upper right corner
    pic3 = GridImg[550:1080, 0:720]    # lower left corner
    pic4 = GridImg[550:1080, 750:1470] # lower right corner

    # Save the 4 separate pictures
    filename1 = '{}/{}_rThigh_{}.jpg'.format(new_directory, video, save_number*save_interval)
    cv2.imwrite(filename1, pic1)
    filename2 = '{}/{}_domArm_{}.jpg'.format(new_directory, video, save_number*save_interval)
    cv2.imwrite(filename2, pic2)
    filename3 = '{}/{}_lThigh_{}.jpg'.format(new_directory, video, save_number*save_interval)
    cv2.imwrite(filename3, pic3)
    filename4 = '{}/{}_domcalf_{}.jpg'.format(new_directory, video, save_number*save_interval)
    cv2.imwrite(filename4, pic4)
    
    return

def video2Img4Grid(filePath, save_interval=1):
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
        # create a new sub-directory in current directory to store the frame pictures
        new_directory = os.path.join(filePath,'video2img')
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
            print(f"{new_directory} created successfully!")
        
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
                GridImg2Imgs(new_directory, video, save_number, save_interval, picture)
                i = 0
                save_number += 1
            
            frame_number += 1   
    
        print('{} pictures saved in {}'.format((save_number-1)*4, new_directory))
        
    # release resource
    video_capture.release()
    return True

filePath_1GridVideos = 'D:\\downloads\\Piloerection\\video\\1-grid_videos'
video2Img(filePath_1GridVideos, save_interval=1)

filePath_4GridVideos = 'D:\\downloads\\Piloerection\\video\\4-grid_videos'
video2Img4Grid(filePath_4GridVideos, save_interval=1)

