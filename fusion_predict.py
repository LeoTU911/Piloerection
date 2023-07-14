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


def calDissimilarity(benchmark_index, features_List):
    
    dissimilarities = []
    benchmark = features_List[benchmark_index]
    
    for features in features_List:
        dissimilarity = torch.norm(features - benchmark).cpu().numpy()
        dissimilarities.append(dissimilarity)
    
    return dissimilarities
    

def predictOnePart_fusion(frameFile_List, frameFile_Path, data_transform, models_List, device, model_weights):
    # predict one body part
    
    prediction_timeLine = pd.DataFrame({'time': [], 'small': [], 'large': [], 'probability':[], 'intensity':[]})
    
    num_Loop = 0
    total_Loop = len(frameFile_List)
    for frameFile in frameFile_List:
        start_time = time.time() # record the start time
        
        file_Path = os.path.join(frameFile_Path, frameFile)
        
        pred_results = [0, 0, 0]
        for i in range(len(models_List)):
            model_tmp        = models_List[i]
            model_weight_tmp = model_weights[i]
            
            predict, predict_cla = predictImg(file_Path, data_transform, model_tmp, device)
            predict = predict.numpy()
            predict_weighted = [x * model_weight_tmp for x in predict]
            
            pred_results = [x + y for x, y in zip(pred_results, predict_weighted)]
        
        predict_cla = pred_results.index(max(pred_results))
            
        result = orgPredResult(frameFile, predict_weighted, predict_cla)
        
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


def createMultiModels(models_List, model_Predict_Weights_Path, num_classes, device):

    # supported models dict
    supported_Models_Dict = {
    'vit_base' : vit_base_patch16_224,
    'vit_large': vit_large_patch16_224,
    'AlexNet'  : AlexNet,
    'vgg19'    : vgg} 
    
    # get the models' weights
    models_Weights_List = os.listdir(model_Predict_Weights_Path)
    
    # save created models
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
                model_weight_path = os.path.join(model_Predict_Weights_Path, weight_tmp)
                # load model weight
                model.load_state_dict(torch.load(model_weight_path, map_location=device))
                model.eval()
        
        # add created model to save list 
        models_List.append(model)
        
    return models_List
            
                
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
    # sub-model path
    sub_Model_Path = os.path.join(predictionSavePath, args.model_name)
    if not os.path.exists(sub_Model_Path):
        os.mkdir(sub_Model_Path)
    
    # get all the frame image names
    videoFile_List = os.listdir(args.data_path)
    
    # get the specified models list
    models_List = args.model_name_list.split(',')
    models_List = [name.strip() for name in models_List]
    
    # get the models' weights
    model_Predict_Weights_Path = args.model_predict_weights_path
    
    for videoFile in videoFile_List:
        
        print('now predict {}'.format(videoFile))
        
        # Create the model once for each video to avoid using more memory 
        # than the total capacity when predictions on a large number of videos
        # create model
        models_List = createMultiModels(models_List               =models_List,
                                        model_Predict_Weights_Path=model_Predict_Weights_Path,
                                        num_classes               =args.num_classes, 
                                        device                    =device)
        
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
                prediction_timeLine = predictOnePart_fusion(frameFile_List_update, 
                                                            frameFile_Path_update, 
                                                            data_transform, 
                                                            models_List, 
                                                            device,
                                                            model_weights)
                # save as csv file
                save_name = '{}/predict_{}_{}.csv'.format(sub_Model_Path, videoFile, part)
                prediction_timeLine.to_csv(save_name, index=False)
                
                # calculate predict accuracy
                if args.cal_acc == 'True':
                   # concat label file path
                   labelFile_str = '4-grid/{}_{}.csv'.format(videoFile, part)
                   labelFile = os.path.join(args.label_path, labelFile_str)
                   name = videoFile+ '_'+ part
                   calAccuracy(predictFile = save_name, LabelFile = labelFile, name = name)
            
        else:
            prediction_timeLine = predictOnePart_fusion(frameFile_List_update, 
                                                        frameFile_Path_update, 
                                                        data_transform, 
                                                        models_List, 
                                                        device,
                                                        model_weights)
            # save
            save_name = sub_Model_Path+ '/predict_'+ videoFile+ '_.csv'
            prediction_timeLine.to_csv(save_name, index=False)
            
            # calculate predict accuracy
            if args.cal_acc == 'True':
               # concat label file path
               labelFile_str = '1-grid/{}_.csv'.format(videoFile)
               labelFile = os.path.join(args.label_path, labelFile_str)
               calAccuracy(predictFile = save_name, LabelFile = labelFile, name = videoFile)
        
        # Delete each model and release the memory
        for model in models_List:
            del model

        # Clear the list itself
        models_List.clear()
        
        print('{} prediction finished. The result saved in {}'.format(videoFile, save_name))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # The directory of the test dataset
    parser.add_argument('--data_path',  type=str, default="./testData")
    # whether calculate its accuracy
    parser.add_argument('--cal_acc',  type=str, default='True')
    # The directory of the test label files
    parser.add_argument('--label_path',  type=str, default="./testData/file")
    # choose trained models
    parser.add_argument('--model_name_list', type=str, default='vit_base')
    # model prediction weights path
    parser.add_argument('--model_predict_weights_path', type=str, default='vit_base')
    # save path
    parser.add_argument('--save_path', type=str, default='./prediction')
    # number of classes
    parser.add_argument('--num_classes', type=int, default=3)
    # trained weights path
    parser.add_argument('--weights_path', type=str, default='./vit_base_patch16_224.pth')
    # class_indices
    parser.add_argument('--class_indices', type=str, default='./class_indices.json')

    opt = parser.parse_args()

    main(opt)
    
    
###############################################################################

# json_path = 'D:\\downloads\\Piloerection\\class_indices.json'
# with open(json_path, "r") as f:
#     class_indict = json.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    