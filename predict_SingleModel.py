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
from Model.ViT import vit_base_patch16_224,vit_large_patch16_224
from Model.AlexNet import AlexNet
from Model.VGG import vgg19,vgg16
from Model.ResNet import resnet50,resnet101

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
        features = model(img.to(device))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    
    # release resource
    del output
        
    return predict, predict_cla, features


def calDissimilarity(benchmark_index, features_List):
    
    dissimilarities = []
    benchmark = features_List[benchmark_index]
    
    for features in features_List:
        dissimilarity = torch.norm(features - benchmark).cpu().numpy()
        dissimilarities.append(dissimilarity)
    
    return dissimilarities
    


def predictOnePart(frameFile_List, frameFile_Path, data_transform, model, device):
    # predict one body part
    
    prediction_timeLine = pd.DataFrame({'time': [], 'small': [], 'large': [], 'probability':[], 'intensity_old':[]})

    time_Features_list = []
    
    num_Loop = 0
    total_Loop = len(frameFile_List)
    for frameFile in frameFile_List:
        start_time = time.time() # record the start time
        
        file_Path = os.path.join(frameFile_Path, frameFile)
        predict, predict_cla, features = predictImg(file_Path, data_transform, model, device)
        result = orgPredResult(frameFile, predict, predict_cla)
        
        time_Features = [result[0], features]
        time_Features_list.append(time_Features)
        
        # write the predict result
        prediction_timeLine.loc[len(prediction_timeLine)] = result
        
        end_time = time.time()
        time_spent_loop = end_time - start_time
        num_Loop += 1
        estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 500)
      
    # sort the prediction result    
    prediction_timeLine = prediction_timeLine.sort_values(by = 'time', ascending=True)
    sorted_Time_Features_list = sorted(time_Features_list, key=lambda x: x[0])
    
    features_List = [sublist[1] for sublist in sorted_Time_Features_list]
    
    # find the benchmark as the lowest intensity
    benchmark_0_index = prediction_timeLine['intensity_old'].idxmin()
    # calculate dissimilarity
    dissimilarities = calDissimilarity(benchmark_0_index, features_List)
    prediction_timeLine['dissimilarity'] = dissimilarities
    
    
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
    
    # Video-by-Video Prediction
    model_dict = {
    'vit_base': vit_base_patch16_224,
    'vit_large': vit_large_patch16_224,
    'AlexNet': AlexNet,
    'vgg19': vgg19,
    'vgg16':vgg16,
    'resnet50': resnet50,
    'resnet101': resnet101
}
    for videoFile in videoFile_List:
        
        print('now predict {}'.format(videoFile))
        
        # Create the model once for each video to avoid using more memory 
        # than the total capacity when predictions on a large number of videos
        # create model
        if args.model_name in model_dict:
            print(f'train {args.model_name} model')
            model = model_dict[args.model_name](num_classes=args.num_classes).to(device)
        else:
            assert args.model_name in model_dict.keys(), "model not supported, please check it"
            
        # load model weights
        model_weight_path = args.weights
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        
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
                prediction_timeLine = predictOnePart(frameFile_List_update, 
                                                     frameFile_Path_update, 
                                                     data_transform, 
                                                     model, 
                                                     device)
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
            prediction_timeLine = predictOnePart(frameFile_List, 
                                                 frameFile_Path, 
                                                 data_transform, 
                                                 model, 
                                                 device)
            # save
            save_name = sub_Model_Path+ '/predict_'+ videoFile+ '_.csv'
            prediction_timeLine.to_csv(save_name, index=False)
            
            # calculate predict accuracy
            if args.cal_acc == 'True':
               # concat label file path
               labelFile_str = '1-grid/{}_.csv'.format(videoFile)
               labelFile = os.path.join(args.label_path, labelFile_str)
               calAccuracy(predictFile = save_name, LabelFile = labelFile, name = videoFile)
        
        # delete current model and release corresponding resources
        del model
        
        print('{} prediction finished. The result saved in {}'.format(videoFile, save_name))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # The directory of the test dataset
    parser.add_argument('--data_path',  type=str, default="./testData")
    # whether calculate its accuracy
    parser.add_argument('--cal_acc',  type=str, default='True')
    # The directory of the test label files
    parser.add_argument('--label_path',  type=str, default="./testData/file")
    # choose one trained model
    parser.add_argument('--model_name', type=str, default='vit_base')
    # save path
    parser.add_argument('--save_path', type=str, default='./prediction')
    # number of classes
    parser.add_argument('--num_classes', type=int, default=3)
    # trained weights path
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224.pth')
    # class_indices
    parser.add_argument('--class_indices', type=str, default='./class_indices.json')

    opt = parser.parse_args()

    main(opt)
    
    
###############################################################################

# json_path = 'D:\\downloads\\Piloerection\\class_indices.json'
# with open(json_path, "r") as f:
#     class_indict = json.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    