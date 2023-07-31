#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:18:22 2023

@author: Chunhui TU
"""

import pandas as pd
import argparse
import os

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
    
    return acc


def main(args):
    
    # get the name and quantity of all models
    model_Names = os.listdir(args.predict_file_path)
    model_Num   = len(model_Names)
    print('number of models:{}; model list:{}'.format(model_Num, model_Names))
    
    acc_models = []
    # calculate the accuracy one by one
    for model_Name in model_Names:
        print('now calculate the accuracy of {} model'.format(model_Name))
        
        predict_File_Path_of_One_Model = os.path.join(args.predict_file_path, model_Name)
        predict_File_List_of_One_Model = os.listdir(predict_File_Path_of_One_Model)
        
        # delete 'predict_' in the predict file name
        sub_string = 'predict_'
        accs = []
        
        for predict_File in predict_File_List_of_One_Model:
            predict_File_Path = os.path.join(predict_File_Path_of_One_Model, predict_File)
            file_Name = predict_File.replace(sub_string, '')
            label_File_Path = os.path.join(args.label_file_path, file_Name)
            
            acc = calAccuracy(predictFile=predict_File_Path , LabelFile=label_File_Path, name=model_Name)
            accs.append(acc)
        
        # calculate the average accuracy of current model
        acc_avg = sum(accs)/len(accs)
        # concat the accuracy and its model name
        print('model:{}, average accuracy:{}'.format(model_Name, acc_avg))
        acc_model = [acc_avg, model_Name]
        # append the average of accuracy
        acc_models.append(acc_model)
            
    # calculate the weights
    df_acc_model = pd.DataFrame(acc_models)
    df_acc_model.columns = ['average_acuracy', 'model_name']

    total_acc = df_acc_model['average_acuracy'].sum()
    df_acc_model['weight'] = df_acc_model['average_acuracy'] / total_acc
    
    # save the weights file
    save_name = args.save_path+ '/model_weights.csv'
    df_acc_model.to_csv(save_name, index=False)
 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_file_path',   type=str, default="label_path")
    parser.add_argument('--predict_file_path', type=str, default="./predict_path")
    parser.add_argument('--save_path',         type=str, default="./save_path")

    opt = parser.parse_args()

    main(opt)