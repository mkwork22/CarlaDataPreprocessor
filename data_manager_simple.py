import re
import os
import csv
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
import traceback
# WORLD_TO_IMG_SCALE = 1
WORLD_TO_IMG_SCALE = 50
import utility as utils


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class Trajectory_():
    def __init__(self):
        self.poses = []


class Pose3D_():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0


class ObjectInfo_():
    def __init__(self):
        self.object_id = None
        self.seq_id = None
        self.object_type = None
        self.length = 0.0
        self.width = 0.0
        self.height = 0.0
        self.pose = Pose3D_()
        self.map_name = None


class DataManager_():
    def __init__(self):
        self.logfile_name = None
        # self.LOGDATA_ROOT = os.getenv('CARLA_VR_LOGDATA_PATH')
        # self.LOGDATA_ROOT = "/media/kenta/Extreme SSD/dataset/carla_VR"
        # self.LOGDATA_ROOT = "/Users/eweng/Downloads/drones"
        self.LOGDATA_ROOT = "/root/code/traj_tool/CarlaDataPreprocessor/drones"
        self.map_name = None

        self.logdata = None
        self.logdata_all_objects = []
        self.BEVimages = []
        self.RGBimages = []
        self.frames = None
        self.timestamps = None

    def load_csv_data(self, dir):
        data = None
        try:
            df = pd.read_csv(dir, delimiter=' ', header=None)
            data = df.values
            self.map_name = dir.split('/')[-1].split('.')[0]
            status = True
        except:
            try:
                df = pd.read_csv(dir, header=None)
                data = df.values
                self.map_name = dir.split('/')[-1].split('.')[0]
                status = True
            except:
                print("[WARN] Error While Reading : " + dir)
                traceback.print_exc()  # Output system error messages
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
            traceback.print_exc()  # Output system error messages
        return data, status

    def set_logfile_dir(self, target_date, target_dir_name, target_file_name):
        # tmp_list = [target_date, target_dir_name, target_file_name]
        self.logfile_name = self.LOGDATA_ROOT
        # for tmp_name in tmp_list:
        #     if tmp_name is not '':
        #         self.logfile_name = self.logfile_name + '/' + tmp_name
        self.logfile_name = self.logfile_name + '/' + target_file_name

        # self.logfile_name = self.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + '/' + target_file_name
        # self.logfile_name = self.LOGDATA_ROOT + '/' + target_date + target_dir_name + '/' + target_file_name
        # print(self.logfile_name)

    def set_image_data_dir(self, target_date, target_dir_name):
        self.BEVimage_dir_name = self.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + '/bev/'
        self.RGBimage_dir_name = self.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + '/rgb/'

    def set_log_root(self, root):
        self.LOGDATA_ROOT = root

    def store_timestamp_list(self):
        self.frames = np.unique(self.logdata[:, 0])
        self.timestamps = np.unique(self.logdata[:, 1])
        # print(len(self.frames), "::", len(self.timestamps))

    def read_logdata(self, meta=None):
        self.logdata, status = self.load_csv_data(self.logfile_name)
        self.data_size = 0
        for ped_id in np.unique(self.logdata[:, 1]):
            ped_id_num_frames = self.logdata[ped_id == self.logdata[:, 1]].shape[0]
            if ped_id_num_frames > self.data_size:
                self.data_size = ped_id_num_frames
        self.data_size = self.data_size
        self.reshape_logdata(meta)
        print(f"self.data_size: {self.data_size}")
        self.store_timestamp_list()

    def read_image(self, directory):
        idx = 0
        fnum_pre = 0
        images = []
        fnum_array = []
        image_list_mod = []
        image_list = list(sorted(glob.glob(directory + '*.png'), key=natural_keys))

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
        obj.object_id = data[1]  # 600
        obj.object_type = data[2]
        obj.pose.x = data[3]
        obj.pose.y = data[4]
        obj.pose.yaw = data[5]
        if 'vehicle' in obj.object_type:
            obj.length = 4.5*WORLD_TO_IMG_SCALE  # TODO: fix to be larger
            obj.width = 1.8*WORLD_TO_IMG_SCALE
            # obj.length = data[5]
            # obj.width = data[6]
        else:  # is human
            obj.length = 0.5*WORLD_TO_IMG_SCALE
            obj.width = 0.5*WORLD_TO_IMG_SCALE

        if 'human' in obj.object_type:
            obj.map_name = self.map_name#str(data[-1])

        return obj

    def reshape_logdata(self, meta):
        loop_idx = 0
        tmp_container = []

        # === Construct object information ===
        for data in self.logdata:
            # Find ego-avatar
            if data[2] == 'human':
                if data[0] != loop_idx:
                    # Construct current object data to logdata_all_objects
                    self.logdata_all_objects.append(tmp_container)
                    tmp_container = []

                    # Update loop_idx
                    loop_idx = data[0]
                    tmp_container.append(self.reconstruct_object_info(data))
            else:
                # Find current objects
                if data[0] == loop_idx:
                    tmp_container.append(self.reconstruct_object_info(data))
