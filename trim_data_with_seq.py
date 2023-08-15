#coding:utf-8

import os
import sys
import time
import cv2
import csv
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

sys.path.append('../')
import data_manager

import utility as utils

MIN_SUMMARY_CSV_EXCEPTION_COL_NUM = 2

def write_csv(fname_with_path, data):
    # Create logfile 
    f = open(fname_with_path, 'w')
    
    # Insert data
    f = open(fname_with_path, 'a')
    writer = csv.writer(f, lineterminator='\n')
    
    for it in data:
        writer.writerow(it)
    f.close()

def load_metadata(DataManager, data_parent_dir):
    dir = data_parent_dir + '/summary.csv'
    meta, status = DataManager.load_csv_data_with_std_func(dir)
    return meta, status

def trimmer(DataManager, meta):
    trimmed_data_list = []
    len_meta = len(meta)
    num_scenes = round((len_meta - MIN_SUMMARY_CSV_EXCEPTION_COL_NUM) / 2)

    for idx in range(num_scenes):
        # Find target indexes
        trimmed_data = DataManager.logdata[(DataManager.logdata[:, 0]>=int(float(meta[idx*2+1]))) & (DataManager.logdata[:,0]<=int(float(meta[idx*2+2])))]
        trimmed_data_list.append(trimmed_data)
    return trimmed_data_list

def add_scenario_info(data, meta):
    mod_data = None
    for it in data:
        if mod_data is None:
            mod_data = np.hstack((it, meta[-1]))
        else:
            mod_data = np.vstack((mod_data, np.hstack((it, meta[-1]))))
    return mod_data

def execute(target_date, target_dir, fpath_list, meta):
    DataManager = data_manager.DataManager_()
    
    for fpath in fpath_list:
        if (meta[0] in fpath):
            target_file_name_with_path = fpath
            target_file_name = fpath.replace(target_dir + '/' + meta[0], "")
            break
     
    savedir = DataManager.LOGDATA_ROOT + '/' + target_date + '/' + meta[0] + '/'
     
    try:
        # Load csv data
        DataManager.logfile_name = target_file_name_with_path
        DataManager.read_logdata()
        
        # Trim raw data according to setting meta file
        trimmed_data_list = trimmer(DataManager, meta)
        print(len(trimmed_data_list))
        
        # Save trimmed data to csv file
        idx=0
        for trimmed_data in trimmed_data_list:
            # DEBUG: Add scenario info
            mod_data = add_scenario_info(trimmed_data, meta)

            print("savedir: ", savedir)
            output_file_name = f'{savedir}trimmed_{idx}.csv'
            write_csv(output_file_name, mod_data)
            idx+=1
        print("[INFO]Finished trimming: ", target_file_name, " \n")
        
        # Deinit
        del DataManager
        
    except Exception as e:
        print("Error occured:", e)
        traceback.print_exc() # Output system error messages

        pass


if __name__ == "__main__":
    DM = data_manager.DataManager_()

    # target_date = '230201'
    target_date = '3230420'
    # target_date = '230427'
    # target_date = '230511'
    # target_date = '230608'
    # target_date = '230724'

    # Obtain csv filename lists
    target_dir = DM.LOGDATA_ROOT + '/' + target_date
    fpath_list = utils.get_csv_file_list(target_dir)
    
    # Load metadata
    metadata, status = load_metadata(DM, target_dir)
    
    for meta in metadata:
        print(meta)
        execute(target_date, target_dir, fpath_list, meta)
    
    # # Multi-processing
    # processes = []
    # with ProcessPoolExecutor(max_workers=20) as executor:
    #     for fpath in fpath_list:
    #         # print("Start multi-processing")
    #         executor.submit(execute, target_date, target_dir_name, target_dir, fpath)