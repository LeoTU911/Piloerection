#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:59:03 2023

@author: Chunhui TU
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def smoothResult(file_Path, smooth_Column_Name, smooth_window, save_Path):
    
    # Read the CSV file
    file_List = os.listdir(file_Path)
    
    # check the save path
    if os.path.exists(save_Path) is False:
        os.makedirs(save_Path)
    
    for file in file_List:
        if file.endswith('.csv'):
            file_with_Path = os.path.join(file_Path,file)
            data = pd.read_csv(file_with_Path)
        
            # get the column
            column = pd.Series(data[smooth_Column_Name])
        
            # smooth
            smoothed_Column = column.rolling(window=smooth_window, center=True).mean()
            
            # return it
            data[smooth_Column_Name] = smoothed_Column
        
            # save the new data
            # get the file's info
            file_Name = file.split('.')[0]
            new_File_Name = '{}_smooth.csv'.format(file_Name) 
            new_File_Path = os.path.join(save_Path, new_File_Name)
            
            data.to_csv(new_File_Path, index=False)
    
    return


def plot_Intensity(file_Path, x_axis, y_axis, save_Path):    
    # Read the CSV file
    file_List = os.listdir(file_Path)
    
    # check the save path
    if os.path.exists(save_Path) is False:
        os.makedirs(save_Path)
    
    for file in file_List:
        if file.endswith('.csv'):
            file_with_Path = os.path.join(file_Path,file)
            df = pd.read_csv(file_with_Path)

            # Filter the DataFrame based on the conditions for coloring
            orange_mask = df['small'] == 1
            red_mask = df['large'] == 1
            blue_mask = (df['small'] == 0) & (df['large'] == 0)
            
            # Set up the figure and axes
            fig, ax = plt.subplots()
            
            # Plot the intensity line
            ax.plot(df[x_axis], df[y_axis], color='blue')
            
            # Color the intensity segments based on the conditions
            ax.fill_between(df[x_axis], df[y_axis], where=orange_mask, color='orange')
            ax.fill_between(df[x_axis], df[y_axis], where=red_mask, color='red')
            ax.fill_between(df[x_axis], df[y_axis], where=blue_mask, color='blue')
            
            # Set the labels and title
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            file_Name = file.split('.')[0]
            ax.set_title('{} Intensity Timeline'.format(file_Name))
            
            
            # Set the y-axis range to [0, 1]
            ax.set_ylim([0, 1])
            
            # Add text description of the icon
            ax.text(0.05, 0.85, 'blue-no goosebumps\norange-small goosebumps\nred-large goosebumps', transform=ax.transAxes, fontsize=8)
            
            # Save the plot as an image file
            save_name = '{}.png'.format(file_Name)
            save_File_Path = os.path.join(save_Path, save_name)
            plt.savefig(save_File_Path, dpi=300) 
            
            # Display the chart
            plt.show()


###############################################################################

file_Path = '/Users/sharp/Desktop/original output'
smooth_save_Path = '/Users/sharp/Desktop/test_output'
smoothResult(file_Path=file_Path, smooth_Column_Name='intensity', smooth_window=31, save_Path=smooth_save_Path)

#file_Path_smooth = '/Users/sharp/Desktop/test_output'
plot_Path = '/Users/sharp/Desktop/test_plot'
plot_Intensity(file_Path=smooth_save_Path, x_axis='time', y_axis='intensity', save_Path=plot_Path)












