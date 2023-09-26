# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:17:00 2023

@author: Chunhui TU
"""

import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # Guaranteed reproducibility of random results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Traversing folders, one folder corresponds to one category
    goosebumps_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sorting to ensure that the order of each platform is consistent
    goosebumps_class.sort()
    # Generate category names and corresponding numeric indices
    class_indices = dict((k, v) for v, k in enumerate(goosebumps_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path  = []  # Store all image paths of the training set
    train_images_label = []  # Store the index information corresponding to the training set pictures
    val_images_path    = []  # Store all image paths of the validation set
    val_images_label   = []  # Store the index information corresponding to the validation set pictures
    every_class_num    = []  # Store the total number of samples for each class
    supported          = [".jpg", ".JPG", ".png", ".PNG"]  # Supported file extension types
    
    # Traverse the files under each folder
    for cla in goosebumps_class:
        cla_path = os.path.join(root, cla)
        # Traverse to get all supported file paths
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sorting to ensure that the order of each platform is consistent
        images.sort()
        # Get the index corresponding to this category
        image_class = class_indices[cla]
        # Record the number of samples for that category
        every_class_num.append(len(images))
        # Proportional random sampling of validation samples
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # If the path is in the sampled validation set, store it in the validation set
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # Otherwise store in the training set
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Anti-Normalize operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove the scale of the x-axis
            plt.yticks([])  # Remove the scale of the y-axis
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # cumulative loss
    accu_num = torch.zeros(1).to(device)   # Cumulative number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Cumulative number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # cumulative loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def estimateTimeConsumption(time_spent_loop, num_Loop, total_Loop, display_interval = 100):
    """
    FUNCTION DESCRIPTION
    Estimate the remaining running time of the program based on 
    the obtained duration of each loop, 
    the total number of loops, 
    and the current number of loops
    
    Parameters
    ----------
    time_spent_loop  : time spent of 1 loop
    num_Loop         : current loop
    total_Loop       : total loop
    display_interval : the time interval of print time info 
                       the default is every 100 loops print once.

    Returns
    -------
    None.
    """
    
    # Display according to the specified interval
    if num_Loop % display_interval == 1:
                    
        estimateTimeLeft = (total_Loop - num_Loop) * time_spent_loop
        ETL_HH = int(estimateTimeLeft//3600)               # Hour
        ETL_MM = int((estimateTimeLeft - ETL_HH*3600)//60) # Minute
        ETL_SS = round(estimateTimeLeft%60, 3)             # Second
        
        print('Current Progress: {}/{}'.format(num_Loop, total_Loop))
        print('Time Spent of 1 Loop: {}s'.format(round(time_spent_loop,4)))
        print('Estimate Time Left: {}:{}:{}'.format('%02d' % ETL_HH, '%02d' % ETL_MM, ETL_SS))
        

     
    
    