import os
import sys
import time
import cv2
import csv
# import rospy
import math
import pickle
import traceback
import argparse
import numpy as np

from data_manager import DataManager_
import utility as utils


class TimeSynclonizerMain_():
    def __init__(self):
        self.simlog_root_dir = f'/media/kenta/Extreme SSD/dataset/carla_VR'
        self.simlog_date = f'/230724'
        self.target_simlog = f'/log_115111'
        self.simlog_dir = f'{self.simlog_root_dir}{self.simlog_date}{self.target_simlog}/' 
        self.image_dir = f'/home/kenta/ego_exo/common/time_synced_exo/01_walk/exo/cam01/images'
        self.simlog_fname = f'logdata_07242023_115111.csv'
        
        self.cam_qr_timesync_result_frame = 5484
        self.cam_qr_timesync_result_timestamp = 1690213790578
        self.cam_freq = 1.0/30.0
        
        self.output_dir = f'{self.simlog_dir}'
        self.output_file = f'timesync_result.csv'
        
        self.sim_to_image_index_list = []
    
    def interp_cam_timestamps(self, max_len):
        frames = [cnt for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        timestamps = [(int(self.cam_qr_timesync_result_timestamp) + \
                       int((self.cam_freq)*1e3)*(cnt-self.cam_qr_timesync_result_frame)) \
                            for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        # diff_timestamps = [int((self.cam_freq)*1e3)*(cnt-self.cam_qr_timesync_result_frame)\
        #                     for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        # print(diff_timestamps[:10])
        return frames, timestamps
    
    def fine_nearest_timestamp(self, target_timestamp, timestamp_list):
        diff_timestamp = timestamp_list - target_timestamp
        index = np.argmin(np.abs(diff_timestamp))
        # print("nearest_index:{}".format(index))
        return self.cam_qr_timesync_result_frame + index
        
    def output_result(self, SimDataHandler):
        result = [SimDataHandler.frames, self.sim_to_image_index_list]
        result_np = np.array(result).T
        np.savetxt(f'{self.output_dir}{self.output_file}', result_np, fmt ='%d')
        
    def execute(self, args):
        # Load sim data
        SimDataHandler = DataManager_()
        SimDataHandler.logfile_name = f'{self.simlog_dir}{self.simlog_fname}'
        SimDataHandler.read_logdata()
        # print(SimDataHandler.timestamps)
        
        # interpolate timestamps in every GoPro images
        image_list = utils.get_jpg_file_list(self.image_dir)
        image_list = [image_path.replace(f'{self.image_dir}/', '').replace('.jpg','') for image_path in image_list]
        # print(image_list)
        frames, interp_timestamps = self.interp_cam_timestamps(int(image_list[-1]))

        # Find nearest frames
        sim_timestamp_pre  = SimDataHandler.timestamps[0]
        for sim_timestamp in SimDataHandler.timestamps:
            sim_timestamp = int(str(sim_timestamp).replace('.', '')[:13])
            # print(sim_timestamp)
            # print("len_sim:{} len_org:{}".format(len(str(sim_timestamp)), len(str(self.cam_qr_timesync_result_timestamp))))
            # print((sim_timestamp - self.cam_qr_timesync_result_timestamp)*1e-3)
            
            target_index = self.fine_nearest_timestamp(sim_timestamp, np.array(interp_timestamps))
            
            # print(target_index)
            
            # time.sleep(1)
            
            # print("time_diff:{}".format((sim_timestamp - sim_timestamp_pre)*1e-3))
            # sim_timestamp_pre = sim_timestamp
            
            self.sim_to_image_index_list.append(target_index)
        
        # print(self.sim_to_image_index_list)
        
        self.output_result(SimDataHandler)
    
    
def parse_args():
  parser = argparse.ArgumentParser(description='Time sync copy')
  parser.add_argument('--sequence', help='')
  parser.add_argument('--cameras', help='')
  parser.add_argument('--frame_rate', help='')
  parser.add_argument('--qr-timestamps', help='these ego timestamps correspond to 00001.jpg to exo')
  parser.add_argument('--sequence-camera-name', help='')
  parser.add_argument('--sequence-start-timestamp', help='')
  parser.add_argument('--sequence-end-timestamp', help='')
  parser.add_argument('--data-dir', help='')
  parser.add_argument('--output-dir', help='')

  args = parser.parse_args()
    
  return args    
    
def main():
    print("Execute QRTimeSynclonizer")
    args = parse_args()
    
    time_syncronizer = TimeSynclonizerMain_()
    time_syncronizer.execute(args)
    

if __name__ == "__main__":
    main()
    