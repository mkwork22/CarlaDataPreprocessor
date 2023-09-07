#coding:utf-8
from __future__ import print_function

import os
import sys
import re
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

scenario_infos = []

class Line:
    x = []
    y = []
    
def set_min_max_values(map, scenario):
    if map in 'Town02_Opt':
        if scenario == '2Lane':
            return 302.0, 307.0
        elif scenario == '2LaneWithParkedCar':
            return 272.5, 276.0
        else:
            return None
    elif map in 'Town04':
        return -250.5, -245.7
    else:
        return None

def extract_frames(raw_data):
    map_name = None
    scenario_name = None
    result_frames = []
    for data in raw_data:
        if data[4] == 'walker.vravatar.egowalker':
            if map_name is None and scenario_name is None:
                map_name = data[-1]
                scenario_name = data[-2]
            x = float(data[5])
            y = float(data[6])
            min_y, max_y = set_min_max_values(data[-1], data[-2])
            if y > min_y and y < max_y:
                # print(data[0])
                result_frames.append(data[0])
                # print(data[-1], data[-2], data[0])
    return result_frames, map_name, scenario_name

def extract_scenario_frames(frames):
    diff_threshold = 10
    buffer_start_frame = 100
    buffer_end_frame = 80
    start_frame = None
    end_frame = None
    start_frames = []
    end_frames = []

    for i in range(1, len(frames)):
        if start_frame is None:
            start_frame = frames[i]
        diff = abs(frames[i] - frames[i-1])
        # print("diff: ", diff)
        if diff >= diff_threshold or i >= len(frames)-1:
            # print(frames[i])
            end_frame = frames[i-1]
            start_frames.append(start_frame-buffer_start_frame)
            end_frames.append(min(end_frame+buffer_end_frame, frames[-1]))
            
            print(start_frame, end_frame)
            # Deinit
            start_frame = None
            end_frame = None
    # print(start_frames, end_frames)

    return start_frames, end_frames

def save_frame_info(start_frames, end_frames, dir_name, map_name, scenario_name):
    print(start_frames, end_frames, dir_name, map_name, scenario_name)
    scenario_name_mod = None
    if scenario_name == '2LaneWithParkedCar':
        scenario_name_mod = 'congesting_road'
    elif scenario_name == '2Lane':
        scenario_name_mod = 'jaywalk'
    elif scenario_name in 'Intersection':
        scenario_name_mod = 'intersection'
    else:
        scenario_name_mod = None

    result = []
    for item1, item2 in zip(start_frames, end_frames):
        result.extend([item1, item2])

    # プレフィックスとサフィックスの追加
    result = [dir_name] + result + [scenario_name_mod]

    # リストを文字列に変換
    result_str = ','.join(map(str, result))

    # 結果の表示
    print(result_str)

    return result_str
    

def execute(target_date, target_dir_name, target_dir, fpath):
    global scenario_infos
    DataManager = data_manager.DataManager_()
    # animator = Animation_()
    target_file_name = fpath.replace(target_dir + '/' + target_dir_name, "")
    dir_name = re.sub(r'/[^/]+\.csv$', '', target_file_name)
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
        all_extracted_frames, map_name, scenario_name = extract_frames(DataManager.logdata)
        start_frames, end_frames = extract_scenario_frames(all_extracted_frames)

        scenario_info = save_frame_info(start_frames, end_frames, dir_name, map_name, scenario_name )
        scenario_infos.append(scenario_info)
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
    fpath_list = utils.get_log_csv_file_list(target_dir)
    
    # Single processing
    for fpath in fpath_list:
        execute(target_date, target_dir_name, target_dir, fpath)

    # global scenario_infos
    for info in scenario_infos:
        print(info)
    
    # # Multi-processing
    # processes = []
    
    # with ProcessPoolExecutor(max_workers=20) as executor:
    #     for fpath in fpath_list:
    #         print("Start multi-processing")
    #         executor.submit(execute, target_date, target_dir_name, target_dir, fpath)
            