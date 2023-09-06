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

def set_parked_vehicle(logdata):
    loop_idx = 0
    parked_vehicle_data = None
    fixed_data = []

    # === Construct object information ===
    for data in logdata:   
        # Find ego-avatar
        if data[4]=='walker.vravatar.egowalker':
            fixed_data.append(data) # Add walker agent

            # Set parking vehicle info
            parked_vehicle_data = data.copy()
            parked_vehicle_data[3] = 99999
            parked_vehicle_data[4] = 'vehicle.toyota.prius'
            parked_vehicle_data[5] = -0.45
            parked_vehicle_data[6] = 274.7
            parked_vehicle_data[7] = 0.8
            parked_vehicle_data[8] = 0
            parked_vehicle_data[9] = 0
            parked_vehicle_data[10] = 0.0
            parked_vehicle_data[11] = 0.0
            parked_vehicle_data[12] = 0.0
            parked_vehicle_data[13] = 0.0
            parked_vehicle_data[20] = 1.8
            parked_vehicle_data[21] = 4.7
            parked_vehicle_data[22] = 1.6
            # print(parked_vehicle_data)

            # print('parked_vehicle:', parked_vehicle_data)
            fixed_data.append(parked_vehicle_data)
            parked_vehicle_data = None
            # print('stacked_data:', fixed_data)
            # print(fixed_data)
        else:
            fixed_data.append(data)
    return fixed_data

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
        DataManager.read_logdata(meta)
        
        # Trim raw data according to setting meta file
        trimmed_data_list = trimmer(DataManager, meta)
        print(len(trimmed_data_list))
        
        # Save trimmed data to csv file
        idx=0
        for trimmed_data in trimmed_data_list:
            # TODO: Add parked vehicle data
            if meta[-1] in 'congesting_road':
                trimmed_data = set_parked_vehicle(trimmed_data)

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

    # target_date = '230420'
    # target_date = '230427'
    # target_date = '230511'
    # target_date = '230608'
    # target_date = '230724'
    # target_date = '230817'
    target_date = '230821'
    # target_date = '230824'
    # target_date = '230901'


    # Obtain csv filename lists
    target_dir = DM.LOGDATA_ROOT + '/' + target_date
    fpath_list = utils.get_log_csv_file_list(target_dir)

    # print(fpath_list) # Print
    
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