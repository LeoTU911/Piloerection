# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:57:59 2023

@author: Chunhui TU
"""

import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import precision_score, f1_score, confusion_matrix 
from thop import profile


# Models
from Model.ViT import vit_base_patch16_224,vit_large_patch16_224
from Model.AlexNet import AlexNet
from Model.VGG import vgg19,vgg16
from Model.ResNet import resnet50,resnet101



def convert_values_label(row):
    if row['small_x'] == 0 and row['large_x'] == 0:
        return 0
    elif row['small_x'] == 1:
        return 1
    elif row['large_x'] == 1:
        return 2
    
def convert_values_pred(row):
    if row['small_y'] == 0 and row['large_y'] == 0:
        return 0
    elif row['small_y'] == 1:
        return 1
    elif row['large_y'] == 1:
        return 2
    

def cal_EvaluationIndices(label_File_Path, predict_File_Path):
    
    # calculate Precision Rate, F1 Score and Confusion Matrix
    # Among them, Precision Rate, F1 Score 
    # is the average value of all the test sets calculated, 
    # and Confusion Matrix is the cumulative value
    
    # read label files and predict files
    label_Lists = os.listdir(label_File_Path)
    pred_Lists  = os.listdir(predict_File_Path)
    
    # save evaluation
    precisions = []
    f1_scores  = []
    cms        = []
    
    
    for pred in pred_Lists:
        file_Name = pred.split('_', 1)[1]
        if file_Name in label_Lists:
            pred_Path_tmp  = os.path.join(predict_File_Path, pred)
            label_Path_tmp = os.path.join(label_File_Path, file_Name)
            # read 
            pred_tmp  = pd.read_csv(pred_Path_tmp)
            label_tmp = pd.read_csv(label_Path_tmp)
            
            # rename label file's column names
            # Define a dictionary to map old column names to new column names
            new_column_names = {'1': 'small', '3': 'large'}
            # Rename the columns using the rename() method
            label_tmp.rename(columns=new_column_names, inplace=True)
            
            # Merge the dataframes based on the 'time' column
            merged_df = pd.merge(label_tmp, pred_tmp, on='time', how='inner')
            
            # convert
            merged_df['converted_label'] = merged_df.apply(convert_values_label, axis=1)
            merged_df['converted_pred']  = merged_df.apply(convert_values_pred, axis=1)
            
            label_conv = merged_df['converted_label'].tolist()
            pred_conv  = merged_df['converted_pred'].tolist()
            
            # calculate weighted precision
            precision_weighted = precision_score(label_conv, pred_conv, average='weighted')
            # weighted f1 score
            f1 = f1_score(label_conv, pred_conv, average='weighted')
            # cumulative confusion matrix
            cm = confusion_matrix(label_conv, pred_conv)
            # save
            precisions.append(precision_weighted)
            f1_scores.append(f1)
            cms.append(cm)
    
    # itialize the accumulated confusion matrix
    total_cm = np.zeros((3, 3), dtype=int)  
    # Accumulate each confusion matrix
    for cm in cms:
        total_cm[:cm.shape[0], :cm.shape[1]] += cm
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no", "small", "large"],
                yticklabels=["no", "small", "large"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix of piloerection classification")
    plt.savefig("confusion_matrix.png", dpi=326, bbox_inches="tight")
    plt.show()
            
    print('max precision:{}\nmin precision:{}\naverage precision:{}'.format(max(precisions), min(precisions), sum(precisions)/len(precisions)))
    print('max f1 score:{}\nmin f1 score:{}\naverage f1 score:{}'.format(max(f1_scores), min(f1_scores), sum(f1_scores)/len(f1_scores)))
    
    return precisions, f1_scores, total_cm




# Model Evaluation
def createModel(model_Name, num_classes, device):

    # supported models dict
    supported_Models_Dict = {
    'vit_base' : vit_base_patch16_224,
    'vit_large': vit_large_patch16_224,
    'AlexNet'  : AlexNet,
    'vgg19'    : vgg19,
    'vgg16'    : vgg16,
    'resnet50' : resnet50,
    'resnet101': resnet101} 
      
    # matching model name
    if model_Name in supported_Models_Dict:
        model_class = supported_Models_Dict[model_Name]
        model = model_class(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Invalid model name: {model}")
        
    return model


def modelEvaluation(model_Name, input_Size):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = createModel(model_Name, num_classes=3, device=device)
    
    # Create a sample input tensor
    input_tensor = torch.randn(input_Size)
    
    # Calculate FLOPs and parameters using thop.profile
    flops, params = profile(model, inputs=(input_tensor,))

    print(f"Estimated FLOPs: {flops / 1e9:.2f} G")
    print(f"Number of parameters: {params / 1e6:.2f} M")
    
    return True

# input_Size = (1, 3, 224, 224)
# model_Name = 'resnet101'
# modelEvaluation(model_Name, input_Size)

###############################################################################

label_File_Path='C:\\Piloerection\\data\\image\\test\\file'

# different prediction results by different models
# fusion
predict_File_Path='C:\\Piloerection\\data\\prediction_result\\prediction_acc\\fusion_prediction'
vit_base='C:\\Piloerection\\data\\predict_train\\0804\\vit_base'
vit_large='C:\\Piloerection\\data\\predict_train\\0804\\vit_large'
alexnet='C:\\Piloerection\\data\\predict_train\\0804\\AlexNet'
vgg16='C:\\Piloerection\\data\\predict_train\\0804\\vgg16'
vgg19='C:\\Piloerection\\data\\predict_train\\0804\\vgg19'
resnet50='C:\\Piloerection\\data\\predict_train\\0804\\resnet50'
resnet101='C:\\Piloerection\\data\\predict_train\\0804\\resnet101'

precisions, f1_scores, total_cm = cal_EvaluationIndices(label_File_Path, resnet101)















