#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:18:22 2023

@author: Chunhui TU
"""

import pandas as pd
import argparse
import os

def calAccuracy(predictFile, LabelFile, modelName):
    
    predict  = pd.read_csv(predictFile)
    label    = pd.read_csv(LabelFile)
    fileName = LabelFile.split('\\')[-1]
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
    
    print('Prediction Accuracy of {} in {}, Correct:{}, Wrong:{}, Accuracy: {}'.format(modelName, fileName, correct, error, acc))
    
    return acc


def main(args):
    
    # get the name and quantity of all models
    model_Names = os.listdir(args.predict_file_path)
    model_Num   = len(model_Names)
    print('number of models:{}; model list:{}'.format(model_Num, model_Names))
    
    # get the label files' names
    label_Names = os.listdir(args.label_file_path)
    # create dataframe to store the specific accuracy
    column_Names = ['model_name'] + label_Names
    acc_Spec = pd.DataFrame(columns=column_Names)
    
    acc_models = []
    acc_Spec_index = 0
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
            label_File_Name = predict_File.replace(sub_string, '')
            label_File_Path = os.path.join(args.label_file_path, label_File_Name)
            
            acc = calAccuracy(predictFile=predict_File_Path , LabelFile=label_File_Path, modelName=model_Name)
            accs.append(acc)       
            # add result to acc_Spec
            acc_Spec.loc[acc_Spec_index, label_File_Name] = acc
        
        # calculate the average accuracy of current model
        acc_avg = sum(accs)/len(accs)
        # concat the accuracy and its model name
        print('model:{}, average accuracy:{}'.format(model_Name, acc_avg))
        acc_model = [acc_avg, model_Name]
        # append the average of accuracy
        acc_models.append(acc_model)
        # finish current row of acc_Spec
        acc_Spec.loc[acc_Spec_index, 'model_name'] = model_Name
        acc_Spec_index += 1
            
    # calculate the weights
    df_acc_model = pd.DataFrame(acc_models)
    df_acc_model.columns = ['average_acuracy', 'model_name']

    total_acc = df_acc_model['average_acuracy'].sum()
    df_acc_model['weight'] = df_acc_model['average_acuracy'] / total_acc
    
    # save the weights file
    save_name = args.save_path+ '/model_weights.csv'
    df_acc_model.to_csv(save_name, index=False)
    
    # save the specific accuracy result
    if args.specific_acc=='True':
        save_name_spec = args.save_path+ '/model_acc.csv'
        acc_Spec.to_csv(save_name_spec, index=False)
 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_file_path',   type=str, default="./")
    parser.add_argument('--predict_file_path', type=str, default="./")
    parser.add_argument('--save_path',         type=str, default="./")
    parser.add_argument('--specific_acc',      type=str, default="True")

    opt = parser.parse_args()

    main(opt)