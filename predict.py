# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:45:41 2023

@author: Chunhui TU
"""

import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ViT_Model import vit_base_patch16_224 as create_model


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

def orgPredResult(frameFile, predict, predict_cla):
    
    time      = int(frameFile.split('.')[0].split('_')[-1])
    probility = np.around(predict[predict_cla].numpy(), 3)
    
    # large
    if predict_cla == 0:
        result = [time, 0, 1, probility]
    # no
    elif predict_cla == 1:
        result = [time, 0, 0, probility]
    # small
    else:
        result = [time, 1, 0, probility]
        
    return result
    

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
        
        prediction_timeLine = pd.DataFrame({'time': [], 'small': [], 'large': [], 'probility':[]})
        
        # Frame-by-frame picture prediction
        for frameFile in frameFile_List:
            file_Path = os.path.join(frameFile_Path, frameFile)
            predict, predict_cla = predictImg(file_Path, data_transform, model, device)
            result = orgPredResult(frameFile, predict, predict_cla)
            
            # write the predict result
            prediction_timeLine.loc[len(prediction_timeLine)] = result
        
        # save the prediction result
        save_name = predictionSavePath+ '/predict_'+ videoFile+ '.csv'
        prediction_timeLine.to_csv(save_name)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # The root directory of the test dataset
    parser.add_argument('--data_path',  type=str, default="./testData")
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
    
    
    
    
    
# test = '070_dom arm_3.jpg'
# test2 = '070_3.jpg'
# test3 = test.split('.')[0].split('_')[-1]
# test4 = test2.split('.')[0].split('_')[-1]
    
frameFile_Path = 'D:\\downloads\\Piloerection\\testData\\frameImage\\024'
frameFile_List = sorted(os.listdir(frameFile_Path))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    