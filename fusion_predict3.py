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

# Models
from Model.ViT import vit_base_patch16_224
from Model.ViT import vit_large_patch16_224
from Model.AlexNet import AlexNet
from Model.VGG import vgg

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
    
    # release resource
    del output
        
    return predict, predict_cla


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
    
    
def matchingModelandPredictWeight(models_Predict_Weights_df):
    
    models_List = ['vit_base', 'vit_large', 'AlexNet', 'vgg19']
    
    models_Predict_Weights_List = []
    for model in models_List:
        
        # Filter the DataFrame based on 'model_name' column
        filtered_df = models_Predict_Weights_df[models_Predict_Weights_df['model_name'] == model]

        # Access the 'weight' column of the filtered DataFrame
        weight_value = filtered_df['weight'].values[0]
        
        models_Predict_Weights_List.append(weight_value)
    
    return models_Predict_Weights_List


def createMultiModels(models_List, models_Weights_Path, num_classes, device):

    # supported models dict
    supported_Models_Dict = {
    'vit_base' : vit_base_patch16_224,
    'vit_large': vit_large_patch16_224,
    'AlexNet'  : AlexNet,
    'vgg19'    : vgg} 
    
    # get the models' weights list
    models_Weights_List = os.listdir(models_Weights_Path)
    
    # create an empty list for saving the created models
    models_List = []
    
    for model_Name_tmp in models_List:    
        # matching model name
        if model_Name_tmp in supported_Models_Dict:
            model_class = supported_Models_Dict[model_Name_tmp]
            model = model_class(num_classes=num_classes).to(device)
        else:
            raise ValueError(f"Invalid model name: {model}")
            
        # matching model and its weights file
        for weight_tmp in models_Weights_List:
            if model in weight_tmp:
                model_weight_path = os.path.join(models_Weights_Path, weight_tmp)
                # load model weight
                model.load_state_dict(torch.load(model_weight_path, map_location=device))
                model.eval()
        
        # add created model to save list 
        models_List.append(model)
        
    return models_List


def createModel(model_Name, models_Weights_Path, num_classes, device):

    # supported models dict
    supported_Models_Dict = {
    'vit_base' : vit_base_patch16_224,
    'vit_large': vit_large_patch16_224,
    'AlexNet'  : AlexNet,
    'vgg19'    : vgg} 
    
    # get the models' weights list
    models_Weights_List = os.listdir(models_Weights_Path)
      
    # matching model name
    if model_Name in supported_Models_Dict:
        model_class = supported_Models_Dict[model_Name]
        model = model_class(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Invalid model name: {model}")
            
    # matching model and its weights file
    for weight_tmp in models_Weights_List:
        if model_Name in weight_tmp:
            model_weight_path = os.path.join(models_Weights_Path, weight_tmp)
            # load model weight
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()
        
    return model
    

def predictOnePart_fusion(frameFile_List, 
                          frameFile_Path, 
                          data_transform, 
                          device, 
                          models_Weights_Path, 
                          models_Predict_Weights_List, 
                          num_classes):
    
    # predict one body part
    prediction_timeLine = pd.DataFrame({'time': [], 'small': [], 'large': [], 'probability':[], 'intensity':[]})
    
    # create models
    vit_base  = createModel(model_Name='vit_base', models_Weights_Path=models_Weights_Path, num_classes=num_classes, device=device)
    vit_large = createModel(model_Name='vit_large', models_Weights_Path=models_Weights_Path, num_classes=num_classes, device=device)
    alexnet   = createModel(model_Name='AlexNet', models_Weights_Path=models_Weights_Path, num_classes=num_classes, device=device)
    vgg19     = createModel(model_Name='vgg19', models_Weights_Path=models_Weights_Path, num_classes=num_classes, device=device)
    
    num_Loop = 0
    total_Loop = len(frameFile_List)
    for frameFile in frameFile_List:
        start_time = time.time() # record the start time
        file_Path = os.path.join(frameFile_Path, frameFile)
        
        # predict
        pred_results = np.array([0.0, 0.0, 0.0], dtype='float64')
        model_List = [vit_base, vit_large, alexnet, vgg19]
        for i in range(4):
            # get the model and its predict weight
            model_Predict_Weight_tmp = models_Predict_Weights_List[i]
            model_tmp = model_List[i]
            # predict
            predict, predict_cla = predictImg(file_Path, data_transform, model_tmp, device)
            predict = predict.numpy()
            # weighted predict result
            predict_weighted = predict * model_Predict_Weight_tmp
            
            pred_results += predict_weighted
        
        # get the final predict class
        predict_cla = np.argmax(pred_results)
            
        result = orgPredResult(frameFile=frameFile, predict=pred_results, predict_cla=predict_cla)
        
        # write the predict result
        prediction_timeLine.loc[len(prediction_timeLine)] = result
        
        end_time = time.time()
        time_spent_loop = end_time - start_time
        num_Loop += 1
        estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 500)
          
    # sort the prediction result    
    prediction_timeLine = prediction_timeLine.sort_values(by = 'time', ascending=True)
    
    # delete models to release resources
    del vit_base, vit_large, alexnet, vgg19
      
    return prediction_timeLine


def orgPredResult(frameFile, predict, predict_cla):
    
    time        = int(frameFile.split('.')[0].split('_')[-1])
    probability = np.around(predict[predict_cla].tolist(), 3)
    predictList = []
    for i in range(len(predict)):
        predictList.append(float(predict[i].tolist()))
        
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
    
    # create prediction results save directory
    predictionSavePath = args.save_path
    if not os.path.exists(predictionSavePath):
        os.mkdir(predictionSavePath)
    
    # get all the frame image names
    videoFile_List = os.listdir(args.data_path)
    
    # get the models' prediction weights
    models_Predict_Weights_df   = pd.read_csv(args.models_predict_weights)
    models_Predict_Weights_List = matchingModelandPredictWeight(models_Predict_Weights_df)

    for videoFile in videoFile_List:
        
        print('now predict {}'.format(videoFile))
      
        frameFile_Path = os.path.join(args.data_path, videoFile)
        frameFile_List = os.listdir(frameFile_Path)
        
        # 4-grid video
        if len(frameFile_List) <= 4:
            partFile_list = frameFile_List
            # update a deeper frame file path
            for part in partFile_list:
                frameFile_Path_update = os.path.join(frameFile_Path, part)
                frameFile_List_update = os.listdir(frameFile_Path_update)
                # prediction
                prediction_timeLine = predictOnePart_fusion(frameFile_List=frameFile_List_update, 
                                                            frameFile_Path=frameFile_Path_update, 
                                                            data_transform=data_transform, 
                                                            device=device,
                                                            models_Weights_Path=args.models_weights_path,
                                                            models_Predict_Weights_List=models_Predict_Weights_List,
                                                            num_classes=args.num_classes)
                # save as csv file
                save_name = '{}/fusionPredict_{}_{}.csv'.format(predictionSavePath, videoFile, part)
                prediction_timeLine.to_csv(save_name, index=False)
                
                # calculate predict accuracy
                if args.cal_acc == 'True':
                   # concat label file path
                   labelFile_str = '4-grid/{}_{}.csv'.format(videoFile, part)
                   labelFile = os.path.join(args.label_path, labelFile_str)
                   name = videoFile+ '_'+ part
                   calAccuracy(predictFile = save_name, LabelFile = labelFile, name = name)
            
        else:
            predictOnePart_fusion(frameFile_List=frameFile_List, 
                                  frameFile_Path=frameFile_Path, 
                                  data_transform=data_transform, 
                                  device=device,
                                  models_Weights_Path=args.models_weights_path,
                                  models_Predict_Weights_List=models_Predict_Weights_List,
                                  num_classes=args.num_classes)
            # save
            save_name = predictionSavePath+ '/fusionPredict_'+ videoFile+ '_.csv'
            prediction_timeLine.to_csv(save_name, index=False)
            
            # calculate predict accuracy
            if args.cal_acc == 'True':
               # concat label file path
               labelFile_str = '1-grid/{}_.csv'.format(videoFile)
               labelFile = os.path.join(args.label_path, labelFile_str)
               calAccuracy(predictFile = save_name, LabelFile = labelFile, name = videoFile)
        
        print('{} prediction finished. The result saved in {}'.format(videoFile, save_name))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # The directory of the test dataset
    parser.add_argument('--data_path',  type=str, default="./testData")
    # whether calculate its accuracy
    parser.add_argument('--cal_acc',  type=str, default='True')
    # The directory of the test label files
    parser.add_argument('--label_path',  type=str, default="./testData/file")
    # model prediction weights path
    parser.add_argument('--models_predict_weights', type=str, default='vit_base')
    # save path
    parser.add_argument('--save_path', type=str, default='./prediction')
    # number of classes
    parser.add_argument('--num_classes', type=int, default=3)
    # trained models' weights path
    parser.add_argument('--models_weights_path', type=str, default='./vit_base_patch16_224.pth')
    # class_indices
    parser.add_argument('--class_indices', type=str, default='./class_indices.json')

    opt = parser.parse_args()

    main(opt)
    
    
###############################################################################

# json_path = 'D:\\downloads\\Piloerection\\class_indices.json'
# with open(json_path, "r") as f:
#     class_indict = json.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    