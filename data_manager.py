import re
import os
import csv
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image 
import traceback  

import utility as utils

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class Trajectory_():
    def __init__(self):
        self.poses = []

class Pose3D_():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        
class ObjectInfo_():
    def __init__(self):
        self.object_id = None
        self.seq_id = None
        self.timestamp = None
        self.object_type = None
        self.length = 0.0
        self.width = 0.0
        self.height = 0.0
        self.pose = Pose3D_()
        self.vel = Pose3D_()
        self.past_trajectory = Trajectory_()
        self.future_trajectory = Trajectory_()
        self.map_name = None        
        
class DataManager_():
    def __init__(self):
        self.logfile_name = None
        # self.LOGDATA_ROOT = os.getenv('CARLA_VR_LOGDATA_PATH')
        self.LOGDATA_ROOT = "/media/kenta/Extreme SSD/dataset/carla_VR"
        
        self.logdata = None
        self.logdata_all_objects = []
        self.BEVimages = []
        self.RGBimages = []
        self.frames = None
        self.timestamps = None
        
    def load_csv_data(self, dir):
        df = None
        try:
            # df = pd.read_csv(dir, header=None)
            df = pd.read_csv(dir)
            data = df.values
            status = True
            # print(simdata)
        except:
            print("[WARN] Error While Reading : " + dir)
            traceback.print_exc() # Output system error messages
            status = False
        return data, status

    def load_csv_data_with_std_func(self, dir):
        status = False
        try:
            with open(dir) as f:
                reader = csv.reader(f)
                data = [row for row in reader]
                status = True
        except:
            print("[WARN] Error While Reading : " + dir)
            traceback.print_exc() # Output system error messages
        return data, status
    
    def set_logfile_dir(self, target_date, target_dir_name, target_file_name):
        tmp_list = [target_date, target_dir_name, target_file_name]
        self.logfile_name = self.LOGDATA_ROOT
        
        for tmp_name in tmp_list:
            if tmp_name is not '':
                self.logfile_name = self.logfile_name + '/' + tmp_name
        # self.logfile_name = self.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + '/' + target_file_name
        # self.logfile_name = self.LOGDATA_ROOT + '/' + target_date + target_dir_name + '/' + target_file_name
        # print(self.logfile_name)
        
    def set_image_data_dir(self, target_date, target_dir_name):
        self.BEVimage_dir_name = self.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + '/bev/'
        self.RGBimage_dir_name = self.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + '/rgb/'
        
    def store_timestamp_list(self):
        self.frames = np.unique(self.logdata[:,0])
        self.timestamps = np.unique(self.logdata[:,1])
        # print(len(self.frames), "::", len(self.timestamps))
    
    def read_logdata(self):
        print("loading data:", self.logfile_name)
        self.logdata, status = self.load_csv_data(self.logfile_name)
        self.data_size = self.logdata[-1, 0]
        # self.logdata, status = self.load_csv_data_with_std_func(self.logfile_name)
        # self.data_size = self.logdata[-1][0]
        # print("data size:", self.data_size)
        self.reshape_logdata()
        self.store_timestamp_list()
        # for it in self.logdata_all_objects:
        #     print(it)
        
    def read_image(self, directory):
        idx = 0
        fnum_pre = 0
        images = []
        fnum_array = []
        image_list_mod  = []
        image_list = list(sorted(glob.glob(directory + '*.png'), key=natural_keys))
        # knum = int((image_list[-1].replace(directory, '')).replace('.png', ''))
    
        for fname in image_list:
            fnum_array.append(int((fname.replace(directory, '')).replace('.png', '')))
            
        # Regenerate image_list
        for fnum in fnum_array:
            if ((fnum - fnum_pre) > 1):
                image_list_mod.append(image_list[idx - 1])
            image_list_mod.append(image_list[idx])
            fnum_pre = fnum
            idx += 1
            
        for fname in image_list_mod:
            print(fname)
            img = Image.open(fname)
            images.append(img)
        return images
            
    def load_BEV_images(self):
        self.BEVimages = self.read_image(self.BEVimage_dir_name)
        
    def load_RGB_images(self):
        self.RGBimages = self.read_image(self.RGBimage_dir_name)
        
    def reconstruct_object_info(self, data):
        obj = ObjectInfo_()
        obj.seq_id = data[0]
        obj.timestamp = data[1]
        obj.object_id = data[3]
        obj.object_type = data[4]
        obj.pose.x = data[5]
        obj.pose.y = data[6]
        obj.pose.z = data[7]
        obj.pose.pitch = data[8]
        obj.pose.roll = data[9]
        obj.pose.yaw = data[10]

        if 'vehicle' in obj.object_type:
            obj.length = 4.5
            obj.width = 1.8
        else:
            obj.length = 0.5
            obj.width = 0.5

        try:
            obj.vel.x = data[11]
            obj.vel.y = data[12]
            obj.vel.z = data[13]
            obj.length = data[20]
            obj.width = data[21]
            obj.height = data[22]
        except:
            pass
        
        if 'walker.vravatar.egowalker' in obj.object_type:
            obj.map_name = str(data[-1])
            
        return obj
        
    def reshape_logdata(self):
        loop_idx = 0
        tmp_container = []
        
        # === Construct object information ===
        for data in self.logdata:     
            # Find ego-avatar
            if data[4]=='walker.vravatar.egowalker':
                # print(data[0])
                if data[0]!=loop_idx:
                    # Construct current object data to logdata_all_objects
                    self.logdata_all_objects.append(tmp_container)
                    tmp_container = []
                
                    # Update loop_idx
                    loop_idx = data[0]
                    tmp_container.append(self.reconstruct_object_info(data))
            else:
                # Find current objects
                if data[0]==loop_idx:
                    tmp_container.append(self.reconstruct_object_info(data))
