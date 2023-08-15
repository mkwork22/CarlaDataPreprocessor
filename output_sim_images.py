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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.image as mpimg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor

import data_manager
import utility as utils

GENERATE_ANIM_WITH_IMAGE = False

class Line:
    x = []
    y = []

'''
Animation
'''
class Animation_():
    def __init__(self):
        # Data
        self.DataManager = None
        
        # Animation
        self.fig0 = plt.figure(figsize=(9, 6))
        self.fig0.canvas.draw()
        self.fig0.suptitle("ObjectVisualizer")
        self.ax0 = [self.fig0.add_subplot(1, 1, 1),]
        
        if GENERATE_ANIM_WITH_IMAGE:
            self.fig0 = plt.figure(figsize=(10, 6))
            self.fig0.canvas.draw()
            self.fig0.suptitle("ObjectVisualizer")
            self.ax0 = [self.fig0.add_subplot(1, 2, 1),
                        self.fig0.add_subplot(2, 2, 2),
                        self.fig0.add_subplot(2, 2, 4)]

        # Plot planning result
        # self.fig1, self.ax1 = plt.subplots(nrows=4, sharex=True, figsize=(8, 6))
        self.rect_0 = patches.Rectangle((0-4.8/2+1.0, 0-1.8/2), 4.8, 1.8, angle=0, ec='r', fill=True, fc="r")  # ego-vehicle (ROH=1.0mとする)
        self.rect_1 = patches.Rectangle((0-4.8/2, 0-1.8/2), 4.8, 1.8, angle=0, ec='b', fill=False)          # surrounding vehicle

        # Buffering data
        self.nav_sec_time = None
        self.decounter_BEV = 0
        self.map_name = None
        
        # EgoWalker initial data
        self.ego_pos_x, self.ego_pos_y = 0.0, 0.0

    def calc_rectangle_point(self, theta, length, width):
        # print(theta)
        R = np.sqrt(np.power(length, 2) + np.power(width, 2))/2.0
        ang = -np.pi/2.0 + theta + np.arctan2(width/2.0, length/2.0)
        x = R*np.cos(ang)
        y = R*np.sin(ang)
        return x, y
    
    def LPF_matrix(self, ur, tau, dt):
        out_LPF = np.zeros_like(ur)
        for i in range(1, ur.shape[0]):
            out_LPF[i] = out_LPF[i-1] + ((ur[i] - out_LPF[i-1]) * dt / tau)
        return out_LPF
    
    def LPF(self, ur, ur_pre, tau, dt):
        out_LPF = ur_pre + ((ur - ur_pre) * dt / tau)
        return out_LPF
    
    def search_neaerst_idx(self, mat, val):
        idx = np.abs(mat - val).argmin()
        # print(idx)
        return idx
    
    def load_map_lane_info(self, map_name):
        self.lane_info = []
        fname = 'map/map_lane_info_' + map_name + '.pkl'
        with open(fname, mode="rb") as f:
            self.lane_info = pickle.load(f)
        # print(self.lane_info)
        
    def plot_lane_info(self):
        for lane in self.lane_info:
            self.ax0[0].plot(lane.y, lane.x, color='darkslategrey', markersize=1)
        
    """
    For save animation
    """        
    def output_image(self, savedir, iter):
        # print("Saving Animation: " + str(iter) +
        #       "/" + str(self.DataManager.data_size))
        print("\rSaving Image: " + str(iter) +
              "/" + str(self.DataManager.data_size), end="")
        LabelName = ["Lateral"]

        for idx_fig in range(len(LabelName)):
            self.ax0[idx_fig].clear()

        self.fig0.suptitle("ObjectVisualizer" + "\n" + 
                            'Seq:' + str('{:d}'.format(iter)) + "[-]" +
                           "\n"
                           )
        
        
        """
            Geometry information
        """
        self.plot_lane_info()
        
        """
            Object information
        """
        # ===  物標情報 === #
        if (self.DataManager.logdata_all_objects[iter]):
            cnt = 0
            self.obj_rects = [self.rect_1 for _ in range(
                len(self.DataManager.logdata_all_objects[iter]))]

            for obj in self.DataManager.logdata_all_objects[iter]:
                # print(obj.object_type, obj.pose.x, obj.pose.y, obj.pose.yaw)
                # === Title ===
                if (obj.object_id == 600):
                    self.fig0.suptitle("ObjectVisualizer" + "\n" + 
                        'Seq:' + str('{:d}'.format(obj.seq_id)) + "[-]" +
                        "\n"
                        )
                
                # === Animation ===
                if (obj.object_id == 99999):
                    pass
                else:
                    label_name = "Obj" + str(obj.object_id)
                    # pose
                    self.ax0[0].plot(
                        obj.pose.y, obj.pose.x,  ".-b", label=label_name)
                        
                    # future trajectory
                    # if obj.object_type == "vehicle*":
                    if 'vehicle' in obj.object_type:
                        idx = 0
                       
                        # rectangle
                        # Calculate rectangle origin
                        rect_origin_x = (-obj.length/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                                        - (-obj.width/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                                        + obj.pose.x
                        rect_origin_y = (-obj.length/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                                        + (-obj.width/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                                        + obj.pose.y
                        self.ax0[0].add_patch(patches.Rectangle((rect_origin_y, rect_origin_x),
                                                                 obj.width, obj.length, 
                                                                 angle=-obj.pose.yaw, 
                                                                 ec='b', fill=False))
                    elif 'walker' in obj.object_type:
                        if obj.object_id == 600:
                            self.ego_pos_x = obj.pose.x
                            self.ego_pos_y = obj.pose.y
                            # print("Obtained VR ego-walker object information", self.ego_pos_x, self.ego_pos_y)
                            
                        self.ax0[0].plot(obj.pose.y, obj.pose.x,  ".-r", label=label_name)
                        # rectangle
                        width, length = obj.width, obj.length
                        # Calculate rectangle origin
                        rect_origin_x = (-obj.length/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                                        - (-obj.width/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                                        + obj.pose.x
                        rect_origin_y = (-obj.length/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                                        + (-obj.width/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                                        + obj.pose.y
                        self.ax0[0].add_patch(patches.Rectangle((rect_origin_y, rect_origin_x),
                                                                 width, length, 
                                                                 angle=-obj.pose.yaw,
                                                                 ec='r', fill=False))
                        # Draw arrow
                        arrow_length = 3.0
                        dy = arrow_length * np.sin(obj.pose.yaw*np.pi/180.0)
                        dx = arrow_length * np.cos(obj.pose.yaw*np.pi/180.0)
                        self.ax0[0].arrow(obj.pose.y, obj.pose.x, dy, dx, head_width=0.05, fc='r', ec='r')
                    else:
                        idx = 0
                        prob_cut_in = False
                        fc_color = 'b'
                        # rectangle
                        width, length = obj.width, obj.length
                        # Calculate rectangle origin
                        rect_origin_x = (-obj.length/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                                        - (-obj.width/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                                        + obj.pose.x
                        rect_origin_y = (-obj.length/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                                        + (-obj.width/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                                        + obj.pose.y
                        self.ax0[0].add_patch(patches.Rectangle((rect_origin_y, rect_origin_x),
                                                                 width, length, 
                                                                angle=-obj.pose.yaw, ec='b', 
                                                                fill=prob_cut_in, fc=fc_color))
                        
                        if (obj.object_id == 2612):
                            print(obj.bbox_bottom_right)
                        
                        # self.obj_rects[cnt] = patches.Rectangle((obj.pose.y - obj.width/2, obj.pose.x - obj.length/2),
                        #                                         obj.width, obj.length, angle=0, ec='b', fill=False)

                    # Information
                    self.ax0[0].text(obj.pose.y, obj.pose.x, 
                                        (" ObjID:" + str(int(obj.object_id)) 
                                        + "\n Loc:[" + str('{:.2f}'.format(obj.pose.x)) + ", " + str('{:.2f}'.format(obj.pose.y)) + "] [m]" +
                                        "\n Yaw:[" + str('{:.2f}'.format(obj.pose.yaw)) + "] [deg]" + "\n"
                                    # "\n StopFlag:" + str(obj.stop_vehicle_flag) + ""
                                        ))
                    # plt.pause(0.000001)

                # # Plot Figure (Rectangle)
                # self.plot0 = self.ax0[0].add_patch(self.rect_0)
                # for it in self.obj_rects:
                #     self.ax0[0].add_patch(it)
                
        
        # self.ax0[0].add_patch(rect_ego)
        # self.ax0[0].plot(traj[:, 1], traj[:, 0], ".-r", label="wide path")
        self.ax0[0].set_xlim(self.ego_pos_y-10.0, self.ego_pos_y+10.0)
        self.ax0[0].set_ylim(self.ego_pos_x-10.0, self.ego_pos_x+10.0)
        self.ax0[0].grid(True, zorder=10)
        self.ax0[0].set_xlabel("Lateral position[m]")
        self.ax0[0].set_ylabel("Longitudinal position[m]")
        self.ax0[0].set_aspect("equal")
        
        plt.savefig(f'{savedir}/{iter:05d}.png', format="png", dpi=300)
        # plt.pause(1)
        
    def get_map_name(self, data):
        map_name = None
        for obj in data:
            if obj.object_id == 600:
                map_name = obj.map_name
                print("map_name: ", map_name)
                break
        return map_name
        
    def save_animation(self, savedir, DataManager):
        # Set DataManager information
        self.DataManager = DataManager
        self.map_name = self.get_map_name(self.DataManager.logdata_all_objects[1])
        self.load_map_lane_info(self.map_name)        # ### BEV Image ###

        # Loop animation
        # ani = animation.FuncAnimation(
        #     self.fig0, self.update_anim, interval=1, frames=DataManager.data_size)
        print("save dir:", savedir)

        for frame in range(DataManager.data_size):
            self.output_image(savedir, frame)
    
    
def execute(target_date, target_dir_name, target_dir, fpath):
    # === Create instance ===
    DataManager = data_manager.DataManager_()
    animator = Animation_()
    target_file_name = fpath.replace(target_dir + '/' + target_dir_name, "")
    # print(target_file_name)
    file_name_list = target_file_name.split('/')
    savedir_supplement = ""
    for i in range(len(file_name_list)):
        if (i >= len(file_name_list)-1):
            pass
        else:
            savedir_supplement = savedir_supplement + file_name_list[i] + '/'
            
    # === Make output directory ===
    output_dir = f'{target_dir}/{target_dir_name}{savedir_supplement}image/'
    os.makedirs(output_dir, exist_ok=True)
    
    # === Load timesync info ===
    timesync_info_dir = f'{target_dir}/{target_dir_name}{savedir_supplement}'
    timesync_info = np.loadtxt(f'{timesync_info_dir}timesync_result.csv')
    
    print(timesync_info)
            
    # === Load data and generate images ===
    try:
        DataManager.set_logfile_dir(target_date, target_dir_name, target_file_name)
        DataManager.set_image_data_dir(target_date, target_dir_name)
        DataManager.read_logdata()
        # DataManager.load_BEV_images()
        # DataManager.load_RGB_images()
                
        # Generate image
        animator.save_animation(output_dir, DataManager)
        print("\n[INFO]Finished genrating animation:", fpath, " \n")
        
        # Deinit
        del DataManager, animator
        
    except Exception as e:
        print("Error occured:", e)
        pass
        
if __name__ == "__main__":
    DataManager = data_manager.DataManager_()

    # target_date = '230201'
    # target_date = '230420'
    # target_date = '230427'
    # target_date = '230511'
    # target_date = '230608'
    target_date = '230724'
    target_dir_name = ''
    
    # Obtain csv filename lists
    target_dir = DataManager.LOGDATA_ROOT + '/' + target_date
    fpath_list = utils.get_csv_file_list(target_dir)
    
    # TODO:Debug:
    fpath = f'/media/kenta/Extreme SSD/dataset/carla_VR/230724/log_115111/logdata_07242023_115111.csv'
    execute(target_date, target_dir_name, target_dir, fpath)
    
    
    # # Single processing
    # for fpath in fpath_list:
    #     execute(target_date, target_dir_name, target_dir, fpath)
    
    # # Multi-processing
    # processes = []
    
    # with ProcessPoolExecutor(max_workers=20) as executor:
    #     for fpath in fpath_list:
    #         print("Start multi-processing")
    #         executor.submit(execute, target_date, target_dir_name, target_dir, fpath)
            