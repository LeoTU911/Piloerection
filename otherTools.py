# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:17:00 2023

@author: Chunhui TU
"""

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
    
    estimateTimeLeft = (total_Loop - num_Loop) * time_spent_loop
    ETL_HH = int(estimateTimeLeft//3600)   # Hour
    ETL_MM = int(estimateTimeLeft//60)     # Minute
    ETL_SS = round(estimateTimeLeft%60, 3) # Second
    
    # Display according to the specified interval
    if num_Loop % display_interval == 1:
        print('Current Progress: {}/{}'.format(num_Loop, total_Loop))
        print('Time Spent of 1 Loop: {}s'.format(round(time_spent_loop,4)))
        print('Estimate Time Left: {}:{}:{}'.format('%02d' % ETL_HH, '%02d' % ETL_MM, ETL_SS))