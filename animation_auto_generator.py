#coding:utf-8
from __future__ import print_function

import os
import pickle
import argparse
import subprocess
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from concurrent.futures import ProcessPoolExecutor

import data_manager
from data_manager_simple import DataManager_ as DataManagerSimple
from data_manager_simple import WORLD_TO_IMG_SCALE

import utility as utils


class Lane:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    x = []
    y = []


def rotate_point(point, angle, pivot):
    x, y = point[0] - pivot[0], point[1] - pivot[1]
    x_new = x * np.cos(angle) - y * np.sin(angle)
    y_new = x * np.sin(angle) + y * np.cos(angle)
    return (x_new + pivot[0], y_new + pivot[1])


'''
Animation
'''
class Animation_():
    def __init__(self, args):
        # Data
        self.DataManager = None
        self.args = args
        
        # Animation
        self.fig0 = plt.figure(figsize=(9, 6))
        self.fig0.canvas.draw()
        self.fig0.suptitle("")#ObjectVisualizer")
        self.ax0 = [self.fig0.add_subplot(1, 1, 1),]

        # Random axis rotation
        self.random_rotation = np.random.uniform(0, 2 * np.pi)
        print(f"self.random_rotation: {self.random_rotation}")

        # Plot planning result
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
        # print(f"map_name: {map_name}")
        fname = 'map/map_lane_info_' + "_".join(map_name.split('-')[0].split("_")[:2]) + '.pkl'
        with open(fname, mode="rb") as f:
            self.lane_info = pickle.load(f)

    def plot_lane_info(self, center_x=None, center_y=None):
        for lane in self.lane_info:
            if center_x is not None and center_y is not None:
                y, x = rotate_point((lane.y, lane.x), self.random_rotation, (center_x, center_y))
            else:
                y, x = lane.y, lane.x
            self.ax0[0].plot(y, x, color='darkslategrey', markersize=1)
        
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

        # self.fig0.suptitle("ObjectVisualizer" + "\n" +
        #                     'Seq:' + str('{:d}'.format(iter)) + "[-]" +
        #                    "\n"
        #                    )
        self.fig0.suptitle(f"\nFrame: {iter}")#\nRandom Rot: {np.rad2deg(self.random_rotation):0.2f} degs from x-axis")

        """
            Object information
        """

        # fixed plotting bounds
        if args.vr:
            self.ax0[0].set_xlim(295, 315)
            self.ax0[0].set_ylim(103, 123)
        else:
            self.ax0[0].set_xlim(400, 1400)
            self.ax0[0].set_ylim(1000, 2000)
            # self.ax0[0].set_xlim(0, 1750)
            # self.ax0[0].set_ylim(500, 2250)

        # Get plotting bounds
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Calculate the center point for rotation
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2

        """
            Geometry information
        """
        self.plot_lane_info(center_x, center_y)

        # ===  物標情報 === #
        if len(self.DataManager.logdata_all_objects) > iter and (self.DataManager.logdata_all_objects[iter]):
            self.obj_rects = [self.rect_1 for _ in range(len(self.DataManager.logdata_all_objects[iter]))]

            for obj in self.DataManager.logdata_all_objects[iter]:
                # === Title ===
                # if (obj.object_id == 600):
                #     self.fig0.suptitle("ObjectVisualizer" + "\n" +
                #         'Seq:' + str('{:d}'.format(obj.seq_id)) + "[-]" +
                #         "\n"
                #         )
                
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
                    # self.ax0[0].plot(
                    #     obj.pose.y, obj.pose.x,  ".-b", label=label_name)
                        
                    # future trajectory
                    # if obj.object_type == "vehicle*":
                    if 'vehicle' in obj.object_type:
                        new_yaw = obj.pose.yaw + -np.rad2deg(self.random_rotation)
                        new_pos_y, new_pos_x = rotate_point((obj.pose.y, obj.pose.x), self.random_rotation, (center_x, center_y))
                        # Calculate rectangle origin
                        # rect_origin_x = (-obj.length/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                        #                 - (-obj.width/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                        #                 + obj.pose.x
                        # rect_origin_y = (-obj.length/2) * np.sin(obj.pose.yaw*np.pi/180.0) \
                        #                 + (-obj.width/2) * np.cos(obj.pose.yaw*np.pi/180.0) \
                        #                 + obj.pose.y
                        rect_origin_x = (-obj.length/2) * np.cos(new_yaw*np.pi/180.0) \
                                        - (-obj.width/2) * np.sin(new_yaw*np.pi/180.0) \
                                        + new_pos_x
                        rect_origin_y = (-obj.length/2) * np.sin(new_yaw*np.pi/180.0) \
                                        + (-obj.width/2) * np.cos(new_yaw*np.pi/180.0) \
                                        + new_pos_y

                        self.ax0[0].add_patch(patches.Rectangle((rect_origin_y, rect_origin_x),
                                                                 obj.width, obj.length, 
                                                                 angle=-new_yaw,
                                                                 # angle=-obj.pose.yaw,
                                                                 ec='b', fill=False))

                    elif 'walker' in obj.object_type or 'human' in obj.object_type:
                        if obj.object_id == 600 or obj.object_type == 'human':
                            self.ego_pos_x = obj.pose.x
                            self.ego_pos_y = obj.pose.y

                        new_yaw = obj.pose.yaw + -np.rad2deg(self.random_rotation)
                        new_pos_y, new_pos_x = rotate_point((obj.pose.y, obj.pose.x), self.random_rotation, (center_x, center_y))

                        # text label for vehicle
                        # self.ax0[0].plot(obj.pose.y, obj.pose.x,  ".-r", label=label_name)
                        # rectangle
                        width, length = obj.width, obj.length
                        # Calculate rectangle origin
                        rect_origin_x = (-obj.length/2) * np.cos(new_yaw*np.pi/180.0) \
                                        - (-obj.width/2) * np.sin(new_yaw*np.pi/180.0) \
                                        + new_pos_x
                        rect_origin_y = (-obj.length/2) * np.sin(new_yaw*np.pi/180.0) \
                                        + (-obj.width/2) * np.cos(new_yaw*np.pi/180.0) \
                                        + new_pos_y
                        self.ax0[0].add_patch(patches.Rectangle((rect_origin_y, rect_origin_x),
                                                                 width, length,
                                                                 angle=-new_yaw,
                                                                 ec='r', fill=False))
                        # Draw arrow for yaw
                        arrow_length = .6
                        if not self.args.vr:
                            arrow_length *= WORLD_TO_IMG_SCALE
                        # dy = arrow_length * np.sin(obj.pose.yaw*np.pi/180.0)
                        # dx = arrow_length * np.cos(obj.pose.yaw*np.pi/180.0)
                        dy = arrow_length * np.sin(new_yaw*np.pi/180.0)
                        dx = arrow_length * np.cos(new_yaw*np.pi/180.0)
                        self.ax0[0].arrow(new_pos_y, new_pos_x, dy, dx, head_width=0.05, fc='r', ec='r')
                        # self.ax0[0].arrow(obj.pose.y, obj.pose.x, dy, dx, head_width=0.05, fc='r', ec='r')
                    else:  # not human nor vehicle
                        import ipdb; ipdb.set_trace()
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
                        rect_origin_y, rect_origin_x = rotate_point((rect_origin_y, rect_origin_x),
                                                                    self.random_rotation, (center_x, center_y))

                        self.ax0[0].add_patch(patches.Rectangle((rect_origin_y, rect_origin_x),
                                                                 width, length, 
                                                                angle=-obj.pose.yaw, ec='b',
                                                                # angle=-new_yaw, ec='b',
                                                                fill=prob_cut_in, fc=fc_color))

                    # human text information
                    # self.ax0[0].text(obj.pose.y, obj.pose.x,
                    #                     (" ObjID:" + str(int(obj.object_id))
                    #                     + "\n Loc:[" + str('{:.2f}'.format(obj.pose.x)) + ", " + str('{:.2f}'.format(obj.pose.y)) + "] [m]" +
                    #                     "\n Yaw:[" + str('{:.2f}'.format(obj.pose.yaw)) + "] [deg]" + "\n"
                    #                 "\n StopFlag:" + str(obj.stop_vehicle_flag) + ""
                                        # ))

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

        # # ped-centered plotting bounds
        # offset = 10.0  # the offset for plotting bounds
        # if not self.args.vr:
        #     offset *= WORLD_TO_IMG_SCALE
        # # self.ax0[0].plot(traj[:, 1], traj[:, 0], ".-r", label="wide path")
        # self.ax0[0].set_xlim(self.ego_pos_y-offset, self.ego_pos_y+offset)
        # self.ax0[0].set_ylim(self.ego_pos_x-offset, self.ego_pos_x+offset)


        # self.ax0[0].grid(True, zorder=10)
        # self.ax0[0].set_xlabel("Lateral position[m]")
        # self.ax0[0].set_ylabel("Longitudinal position[m]")
        self.ax0[0].set_xlabel("")
        self.ax0[0].set_ylabel("")
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

        # ticks
        self.ax0[0].set_xticklabels([])
        self.ax0[0].set_yticklabels([])

        # ### RGB Spectator Image ###
        # self.ax0[2].clear()
        # if (self.DataManager.RGBimages is not None):
        #     if (self.DataManager.RGBimages[iter] is not None):
        #         self.ax0[2].imshow(self.DataManager.RGBimages[iter])
        #     else:
        #         self.ax0[2].imshow(self.RGBimg_backup)
        #     self.ax0[2].set_xticklabels([])
        #     self.ax0[2].set_yticklabels([])
        
    def get_map_name(self, data, data_manager):
        if hasattr(data_manager, 'map_name') and data_manager.map_name is not None:
            return data_manager.map_name
        map_name = None
        for obj in data:
            if obj.object_id == 600:
                map_name = obj.map_name
                break
        return map_name
        
    def save_animation(self, savedir, DataManager):
        # Set DataManager information
        self.DataManager = DataManager
        self.map_name = self.get_map_name(self.DataManager.logdata_all_objects[1], DataManager)
        self.load_map_lane_info(self.map_name)
        
        # Loop animation
        ani = animation.FuncAnimation(self.fig0, self.update_anim, interval=1, frames=DataManager.data_size)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        # Save animation (Requirement:ffmpeg (You can get by "sudo apt install ffmpeg")))
        fps = 20
        if self.args.vr:
            anim_filename = f"{savedir}{self.map_name}-{DataManager.participant_name}.mp4"
            gif_filename = f"{savedir}{self.map_name}-{DataManager.participant_name}.gif"
            print(f"saving animation to {anim_filename}")
            ani.save(anim_filename, writer="ffmpeg", fps=fps)
        else:
            anim_filename = f"{savedir}{self.map_name}.mp4"
            gif_filename = f"{savedir}{self.map_name}.gif"
            print(f"saving animation to {anim_filename}")
            ani.save(anim_filename, writer="ffmpeg", fps=fps)
        print(f"saved animation to {anim_filename}")

        mp4_to_gif(anim_filename, gif_filename, fps)
        # ani.save(savedir+"animation.mp4", writer="ffmpeg", fps=20)

def mp4_to_gif(input_file, output_file, fps):
    command = ["ffmpeg", "-y", "-i", input_file, "-vf", f"fps={fps},scale=720:-1", "-c:v", "gif", output_file]
    subprocess.run(command)
    print(f"saved gif to {output_file}")


def execute(target_date, target_dir_name, target_dir, fpath, args=None):
    if args.vr:
        DataManager = data_manager.DataManager_()
    else:
        DataManager = DataManagerSimple()
        DataManager.set_log_root(target_dir)
    animator = Animation_(args)
    target_file_name = fpath.replace(target_dir + '/' + target_dir_name, "")
    file_name_list = target_file_name.split('/')
    savedir_supplement = ""
    for i in range(len(file_name_list)):
        if (i >= len(file_name_list)-1):
            pass
        else:
            savedir_supplement = savedir_supplement + file_name_list[i] + '/'
    # try:
    DataManager.set_logfile_dir(target_date, target_dir_name, target_file_name)
    # DataManager.set_image_data_dir(target_date, target_dir_name)
    DataManager.read_logdata()
    # DataManager.load_BEV_images()
    # DataManager.load_RGB_images()

    # savedir = '~/Downloads/drones/anim_vr/' + target_dir_name + savedir_supplement
    # savedir = DataManager.LOGDATA_ROOT + '/' + target_date + '/' + target_dir_name + savedir_supplement

    if args.vr:
        savedir = '/root/code/traj_tool/CarlaDataPreprocessor/drones/trajs_up/anim_vr/'
    else:
        savedir = DataManager.LOGDATA_ROOT + '/anim_real/'
    animator.save_animation(savedir, DataManager)

    # Deinit
    del DataManager, animator

    # except Exception as e:
    #     traceback.print_exc()
    #     print("Error occured:", e)

        
if __name__ == "__main__":
    # target_date = '230201'
    # target_date = '230420'
    # target_date = '230427'
    # target_date = '230511'
    # target_date = '230608'
    # target_date = '230724'
    # target_date = '230817' 
    # target_date = '08212023'
    # target_date = '08242023'  
    target_dir_name = ''
    # Obtain csv filename lists

    parser = argparse.ArgumentParser()
    # parser.add_argument('--remote_vr', action='store_true', default='')
    parser.add_argument('--map_name', action='store_true', default='')
    parser.add_argument('--vr', action='store_true', default='')
    args = parser.parse_args()

    if args.vr:
        DataManager = data_manager.DataManager_()
    else:
        DataManager = DataManagerSimple()

    # VR
    if args.vr:
        target_date = '09012023'
        # target_date = 'intersection'
        target_dir = DataManager.LOGDATA_ROOT + '/' + target_date
        # target_dir = 'root/code/traj_tool/CarlaDataPreprocessor/logdata'
    else:
        target_date = 'trajs_up'
        target_dir = DataManager.LOGDATA_ROOT + '/' + target_date
        # target_dir = '/Users/eweng/code/traj_tool/CarlaDataPreprocessor/logdata'
        DataManager.set_log_root(target_dir)

    fpath_list = utils.get_csv_file_list(target_dir)
    print(f"fpath_list: {fpath_list}")
    print(f"len(fpath_list): {len(fpath_list)}")

    # Single processing
    # for fpath in fpath_list:
    #     execute(target_date, target_dir_name, target_dir, fpath, args)
    #     exit()
    # # Multi-processing
    processes = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        for fpath in fpath_list:
            print("Start multi-processing")
            # executor.submit(execute, target_date, target_dir_name, target_dir, fpath)
            executor.submit(execute, target_date, target_dir_name, target_dir, fpath, args)
