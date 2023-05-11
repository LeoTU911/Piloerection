# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:45:41 2023

@author: Chunhui TU
"""

import os
import json
import argparse
import time

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ViT_Model import vit_base_patch16_224 as create_model
from utils import estimateTimeConsumption


# Single Frame Image Prediction
def predictImg(file_Path, data_transform, model, device):
    
    # load image
    assert os.path.exists(file_Path), "file: '{}' dose not exist.".format(file_Path)
    img = Image.open(file_Path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        
    return predict, predict_cla

def predictOnePart(frameFile_List, frameFile_Path, data_transform, model, device):
    # predict one body part
    
    prediction_timeLine = pd.DataFrame({'time': [], 'small': [], 'large': [], 'probability':[], 'intensity':[]})
    
    num_Loop = 0
    total_Loop = len(frameFile_List)
    for frameFile in frameFile_List:
        start_time = time.time() # record the start time
        
        file_Path = os.path.join(frameFile_Path, frameFile)
        predict, predict_cla = predictImg(file_Path, data_transform, model, device)
        result = orgPredResult(frameFile, predict, predict_cla)
        
        # write the predict result
        prediction_timeLine.loc[len(prediction_timeLine)] = result
        
        end_time = time.time()
        time_spent_loop = end_time - start_time
        num_Loop += 1
        estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 500)
      
    # sort the prediction result    
    prediction_timeLine = prediction_timeLine.sort_values(by = 'time', ascending=True)
    
    return prediction_timeLine

def orgPredResult(frameFile, predict, predict_cla):
    
    time        = int(frameFile.split('.')[0].split('_')[-1])
    probability   = np.around(predict[predict_cla].numpy(), 3)
    predictList = []
    for i in range(len(predict)):
        predictList.append(float(predict[i].numpy()))
        
    # get its intensity
    intensity = calIntensity(predictList)
    
    # large
    if predict_cla == 0:
        result = [time, 0, 1, probability, intensity]
    # no
    elif predict_cla == 1:
        result = [time, 0, 0, probability, intensity]
    # small
    else:
        result = [time, 1, 0, probability, intensity]
        
    return result

def calIntensity(predictList):
    
    # set params
    params = [1, 1e-5, 0.5] # coefficients of large/no/small goosebumps
    intensity = 0           # default intensity
    
    # normalize the probilities of prediction
    for prob in predictList:
        prob = prob / sum(predictList)
        
    # calculate its intensity
    intensity = (np.array(params) * np.array(predictList)).sum()
    intensity = round(intensity, 5)
    
    return intensity
    
def calAccuracy(predictFile, LabelFile, name):
    
    predict = pd.read_csv(predictFile)
    label   = pd.read_csv(LabelFile)
    
    # Rename the columns
    new_names = {"time": "time", "1": "small", "3": "large"}
    label = label.rename(columns=new_names)

    correct = 0
    error   = 0

    for i in range(predict.shape[0]):
        pred_tmp =  predict.loc[i]
        label_tmp = label[label['time'] == pred_tmp['time']]
        
        if pred_tmp['small'] == label_tmp.iloc[0]['small'] and pred_tmp['large'] == label_tmp.iloc[0]['large']:
            correct += 1
        else:
            error += 1

    acc = round((correct / int(predict.shape[0])), 4)
    
    print('Prediction Accuracy of {}, Correct:{}, Wrong:{}, Accuracy: {}'.format(name, correct, error, acc))
    
    
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # read class_indict
    json_path = args.class_indices
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=args.num_classes).to(device)
    # load model weights
    model_weight_path = args.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    
    # create prediction results save directory
    parent_dir = os.path.dirname(args.data_path)
    predictionSavePath = os.path.join(parent_dir, "prediction")
    if not os.path.exists(predictionSavePath):
            os.mkdir(predictionSavePath)
    
    # get all the frame image names
    videoFile_List = os.listdir(args.data_path)
    
    # Video-by-Video Prediction
    for videoFile in videoFile_List:
        
        frameFile_Path = os.path.join(args.data_path, videoFile)
        frameFile_List = os.listdir(frameFile_Path)
        
        # 4-grid video
        if len(frameFile_List) == 4:
            partFile_list = frameFile_List
            # update a deeper frame file path
            for part in partFile_list:
                frameFile_Path_update = os.path.join(frameFile_Path, part)
                frameFile_List_update = os.listdir(frameFile_Path_update)
                # prediction
                prediction_timeLine = predictOnePart(frameFile_List_update, frameFile_Path_update, data_transform, model, device)
                # save as csv file
                save_name = '{}/predict_{}_{}.csv'.format(predictionSavePath, videoFile, part)
                prediction_timeLine.to_csv(save_name, index=False)
                
                # concat label file path
                labelFile_str = '4-grid/{}_{}.csv'.format(videoFile, part)
                labelFile = os.path.join(args.label_path, labelFile_str)
                name = videoFile+ '_'+ part
                
                # calculate predict accuracy
                calAccuracy(predictFile = save_name, LabelFile = labelFile, name = name)
        else:
            prediction_timeLine = predictOnePart(frameFile_List, frameFile_Path, data_transform, model, device)
            # save
            save_name = predictionSavePath+ '/predict_'+ videoFile+ '_.csv'
            prediction_timeLine.to_csv(save_name, index=False)
            
            # concat label file path
            labelFile_str = '1-grid/{}_.csv'.format(videoFile)
            labelFile = os.path.join(args.label_path, labelFile_str)
            
            # calculate predict accuracy
            calAccuracy(predictFile = save_name, LabelFile = labelFile, name = videoFile)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # The directory of the test dataset
    parser.add_argument('--data_path',  type=str, default="./testData")
    # The directory of the test label files
    parser.add_argument('--label_path',  type=str, default="./testData/file")
    # number of classes
    parser.add_argument('--num_classes', type=int, default=3)
    # trained weights path
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224.pth',)
    # class_indices
    parser.add_argument('--class_indices', type=str, default='./class_indices.json',)

    opt = parser.parse_args()

    main(opt)
    
    
###############################################################################

# json_path = 'D:\\downloads\\Piloerection\\class_indices.json'
# with open(json_path, "r") as f:
#     class_indict = json.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    