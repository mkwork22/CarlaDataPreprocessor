#coding:utf-8
from __future__ import print_function

import os
import sys
# sys.path.append('../')
import time
import cv2
# import rospy
import math
import pickle
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.image as mpimg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor

# import analyzer_setting as sets
import data_manager
import utility as utils

class Line:
    x = []
    y = []
    
def extract_frames(raw_data, target_x, target_y):
    result_frames = []
    for data in raw_data:
        print(data)
        if data[4] == 'walker.vravatar.egowalker':
            x = float(data[5])
            y = float(data[6])
            if x == target_x and y == target_y:
                result_frames.append(data[0])
    return result_frames

def execute(target_date, target_dir_name, target_dir, fpath):
    DataManager = data_manager.DataManager_()
    # animator = Animation_()
    target_file_name = fpath.replace(target_dir + '/' + target_dir_name, "")
    # print(target_file_name)
    file_name_list = target_file_name.split('/')
    savedir_supplement = ""
    for i in range(len(file_name_list)):
        if (i >= len(file_name_list)-1):
            pass
        else:
            savedir_supplement = savedir_supplement + file_name_list[i] + '/'
    try:
        DataManager.set_logfile_dir(target_date, target_dir_name, target_file_name)
        # DataManager.set_image_data_dir(target_date, target_dir_name)
        DataManager.read_logdata()

        extract_frames(DataManager.logdata, 280, 290)
        
        # savedir = DataManager.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + savedir_supplement 
        # animator.save_animation(savedir, DataManager)
        # print("\n[INFO]Finished genrating animation:", fpath, " \n")
        
        # Deinit
        del DataManager
        
    except Exception as e:
        traceback.print_exc()
        print("Error occured:", e)
        
if __name__ == "__main__":
    DataManager = data_manager.DataManager_()

    # target_date = '230201'
    # target_date = '230420'
    # target_date = '230427'
    # target_date = '230511'
    # target_date = '230608'
    # target_date = '230724'
    # target_date = '230817' 
    # target_date = '230821'
    # target_date = '230824'  
    target_date = '230901'  
    
    target_dir_name = ''
    
    # Obtain csv filename lists
    target_dir = DataManager.LOGDATA_ROOT + '/' + target_date
    fpath_list = utils.get_csv_file_list(target_dir)
    
    # Single processing
    for fpath in fpath_list:
        execute(target_date, target_dir_name, target_dir, fpath)
    
    # # Multi-processing
    # processes = []
    
    # with ProcessPoolExecutor(max_workers=20) as executor:
    #     for fpath in fpath_list:
    #         print("Start multi-processing")
    #         executor.submit(execute, target_date, target_dir_name, target_dir, fpath)
            