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
        
        # if sets.GENERATE_ANIM_WITH_IMAGE:
        #     self.fig0 = plt.figure(figsize=(10, 6))
        #     self.fig0.canvas.draw()
        #     self.fig0.suptitle("ObjectVisualizer")
        #     self.ax0 = [self.fig0.add_subplot(1, 2, 1),
        #                 self.fig0.add_subplot(2, 2, 2),
        #                 self.fig0.add_subplot(2, 2, 4)]

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
    def update_anim(self, iter):
        # print("Saving Animation: " + str(iter) +
        #       "/" + str(self.DataManager.data_size))
        print("\rSaving Animation: " + str(iter) +
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
                    # # past trajectory
                    # for trajectory in obj.past_trajectory.coordinates:
                    #     self.ax0[0].plot(
                    #         trajectory.y, trajectory.x, marker="o", markersize=2, color="#D0D0F0")
                    # pose
                    self.ax0[0].plot(
                        obj.pose.y, obj.pose.x,  ".-b", label=label_name)
                        
                    # future trajectory
                    # if obj.object_type == "vehicle*":
                    if 'vehicle' in obj.object_type:
                        # print("ID", obj.object_id, "Pos:", obj.pose.x, obj.pose.y)
                        idx = 0
                        # for trajectory in obj.crossing_prediction.trajectories:
                        #     # print(trajectory)
                        #     # for pos in trajectory.coordinates:
                        #     #         self.ax0[0].plot(pos.y, pos.x, ".y")
                        #     if obj.crossing_prediction.probabilities[idx] > 0.16:
                        #         for pos in trajectory.coordinates:
                        #             self.ax0[0].plot(
                        #                 pos.y, pos.x, '.--', color='red', markersize=4)
                        #     elif obj.crossing_prediction.probabilities[idx] > 0.01:
                        #         for pos in trajectory.coordinates:
                         #             self.ax0[0].plot(
                        #                 pos.y, pos.x, '.--', color='#adff2f', markersize=2)
                        #     idx += 1
                            
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
                        # for trajectory in obj.cut_in_prediction.trajectories:
                        #     marker_size = round(obj.cut_in_prediction.probabilities[idx]/2 * 10)
                        #     if obj.cut_in_prediction.probabilities[idx] > 0.5:
                        #         for pos in trajectory.coordinates:
                        #             self.ax0[0].plot(pos.y, pos.x, '.', color='r', markersize=marker_size)
                        #         if (obj.cut_in_prediction.states[idx] == 1):
                        #             prob_cut_in = True
                        #             fc_color = "y"
                        #     elif obj.cut_in_prediction.probabilities[idx] > 0.01:
                        #         for pos in trajectory.coordinates:
                        #             self.ax0[0].plot(pos.y, pos.x, '.', color='#FFD400', markersize=marker_size)                                
                        #     # Text message
                        #     if (obj.cut_in_prediction.states[idx] == 1):
                        #         # Cut-in (Front)
                        #         self.ax0[0].text(trajectory.coordinates[-15].y, trajectory.coordinates[-15].x,
                        #                          (" CutInProb:" + str(float('{:.2f}'.format(
                        #                              obj.cut_in_prediction.probabilities[idx]))))
                        #                          )
                        #     idx += 1
                            
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
                
         
        # """
        #     自車情報
        # """
        # # === 自車情報設定 === #
        # self.ax0[0].text(-1.0, -5.0,
        #                  " EgoVehicle:" +
        #                  "\n V:" +
        #                  str('{:.2f}'.format(
        #                      self.DataManager.vehicle_state[iter].current_speed_mps*3.6)) + "[km/h]"
        #                  )
        # # 自車rectangleプロット
        # ego_ofs_y, ego_ofs_x = self.calc_rectangle_point(0.0, 4.8, 1.8)
        # rect_ego = patches.Rectangle(
        #     (0.0-ego_ofs_y, 0.0+ego_ofs_x), 1.8, 4.8, angle=-0.0*180/np.pi, ec='r', fill=False)
        # # 自車目標軌道生成
        # traj = None
        # speed_traj = None
        # for wp in self.DataManager.wide_path[iter].way_points:
        #     if traj is None:
        #         traj = np.array([[wp.coordinate.x, wp.coordinate.y]])
        #         speed_traj = np.array(
        #             [[wp.arrival_time, wp.velocity, wp.acceleration, wp.curvature]])
        #         # speed_traj = np.array([[wp.arrival_time, wp.velocity, wp.acceleration, abs(min(1.0/max(wp.curvature, 0.0001), 10000.0))]]) # transform Curvature to Radius
        #     else:
        #         traj = np.append(traj, np.array(
        #             [[wp.coordinate.x, wp.coordinate.y]]), axis=0)
        #         speed_traj = np.append(speed_traj, np.array(
        #             [[wp.arrival_time, wp.velocity, wp.acceleration, wp.curvature]]), axis=0)
        #         # speed_traj = np.append(speed_traj, np.array(
        #         #     [[wp.arrival_time, wp.velocity, wp.acceleration, abs(min(1.0/max(wp.curvature, 0.0001), 10000.0))]]), axis=0)
        # self.wide_traj = traj
        
        # self.ax0[0].add_patch(rect_ego)
        # # 自車目標軌道
        # self.ax0[0].plot(traj[:, 1], traj[:, 0], ".-r", label="wide path")
        self.ax0[0].set_xlim(self.ego_pos_y-10.0, self.ego_pos_y+10.0)
        self.ax0[0].set_ylim(self.ego_pos_x-10.0, self.ego_pos_x+10.0)
        self.ax0[0].grid(True, zorder=10)
        self.ax0[0].set_xlabel("Lateral position[m]")
        self.ax0[0].set_ylabel("Longitudinal position[m]")
        self.ax0[0].set_aspect("equal")
        
        # ### BEV Image ###
        # # print(self.DataManager.BEVimages[iter])
        # self.ax0[1].clear()
        # if (self.DataManager.BEVimages is not None):
        #     if (self.DataManager.BEVimages[iter] is None):
        #         self.decounter_BEV -= 1
        #         print("decounter_BEV: ", self.decounter_BEV)
        #     self.ax0[1].imshow(self.DataManager.BEVimages[iter + self.decounter_BEV])
        #     self.ax0[1].set_xticklabels([])
        #     self.ax0[1].set_yticklabels([])
        
        # ### RGB Spectator Image ###
        # self.ax0[2].clear()
        # if (self.DataManager.RGBimages is not None):
        #     if (self.DataManager.RGBimages[iter] is not None):
        #         self.ax0[2].imshow(self.DataManager.RGBimages[iter])
        #     else:
        #         self.ax0[2].imshow(self.RGBimg_backup)
        #     self.ax0[2].set_xticklabels([])
        #     self.ax0[2].set_yticklabels([])
        
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
        self.load_map_lane_info(self.map_name)
        
        # Loop animation
        ani = animation.FuncAnimation(
            self.fig0, self.update_anim, interval=1, frames=DataManager.data_size)
        print("save dir:", savedir)
        # Save animation (Requirement:ffmpeg (You can get by "sudo apt install ffmpeg")))
        ani.save(savedir+"animation.mp4", writer="ffmpeg", fps=20)
    
    
def execute(target_date, target_dir_name, target_dir, fpath):
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
    try:
        DataManager.set_logfile_dir(target_date, target_dir_name, target_file_name)
        DataManager.set_image_data_dir(target_date, target_dir_name)
        DataManager.read_logdata()
        DataManager.load_BEV_images()
        DataManager.load_RGB_images()
        
        savedir = DataManager.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + savedir_supplement 
        animator.save_animation(savedir, DataManager)
        print("\n[INFO]Finished genrating animation:", fpath, " \n")
        
        # Deinit
        del DataManager, animator
        
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
    # target_date = '08212023'
    # target_date = '08242023'  
    target_date = '09012023'  
    
    target_dir_name = ''
    
    # Obtain csv filename lists
    target_dir = DataManager.LOGDATA_ROOT + '/' + target_date
    fpath_list = utils.get_csv_file_list(target_dir)
    
    
    # Single processing
    # for fpath in fpath_list:
    #     execute(target_date, target_dir_name, target_dir, fpath)
    
    
    # Multi-processing
    processes = []
    
    with ProcessPoolExecutor(max_workers=20) as executor:
        for fpath in fpath_list:
            print("Start multi-processing")
            executor.submit(execute, target_date, target_dir_name, target_dir, fpath)
            