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
import settings as sets

class TimeSynclonizerMain_():
    def __init__(self):
        self.simlog_root_dir = sets.SIMDATA_ROOT_DIR
        self.simlog_date = sets.LOGDATE
        self.target_simlog = sets.LOGDIR
        self.simlog_dir = f'{self.simlog_root_dir}{self.simlog_date}{self.target_simlog}/' 
        # self.image_dir = f'/media/kenta/ExtremePro/ego_exo/common/time_synced_exo/01_walk/exo/cam01/images'
        self.image_dir = f'{sets.IMAGE_DATA_ROOT_DIR}{sets.IMAGE_DATA_TARGET_DIR}'
        # self.simlog_fname = f'logdata_08172023_131449.csv'
        self.simlog_fname = sets.SIMLOG_FNAME
        
        # self.cam_qr_timesync_result_frame_start = 5484
        # self.cam_qr_timesync_result_timestamp_start = 1690213790578
        # self.cam_qr_timesync_result_frame_end = None
        # self.cam_qr_timesync_result_timestamp_end = None
        
        # 0818 celine timesync info
        self.cam_qr_timesync_result_frame_start = 3804
        self.cam_qr_timesync_result_timestamp_start = 1692292106638
        self.cam_qr_timesync_result_frame_end = 25218
        self.cam_qr_timesync_result_timestamp_end = 1692292820035
        
        # # 0818 nathaniel timesync info
        # self.cam_qr_timesync_result_frame_start = 7433
        # self.cam_qr_timesync_result_timestamp_start = 1692295119121
        # self.cam_qr_timesync_result_frame_end = 23569
        # self.cam_qr_timesync_result_timestamp_end = 1692295656246

        # # 0818 darren timesync info
        # self.cam_qr_timesync_result_frame_start = 3948
        # self.cam_qr_timesync_result_timestamp_start = 1692297196401
        # self.cam_qr_timesync_result_frame_end = 21405
        # self.cam_qr_timesync_result_timestamp_end = 1692297778278
        
        # # 0818 darren timesync info
        # self.cam_qr_timesync_result_frame_start = 4438
        # self.cam_qr_timesync_result_timestamp_start = 1692298554054
        # self.cam_qr_timesync_result_frame_end = 20847
        # self.cam_qr_timesync_result_timestamp_end = 1692299101340
        
        self.cam_freq_ms = (1.0/30.00000) * 1e3
        
        self.output_dir = f'{self.simlog_dir}'
        self.interp_timestamp = f'interp_timestamp.csv'
        self.output_file = f'timesync_result.csv'
        
        self.sim_to_image_index_list = []
    
    def interp_cam_timestamps(self, max_len):
        intercept = self.cam_qr_timesync_result_timestamp_start - self.cam_freq_ms*self.cam_qr_timesync_result_frame_start
        frames = [cnt for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        timestamps = [int((self.cam_freq_ms)*cnt + intercept) \
                        for cnt in range(self.cam_qr_timesync_result_frame_start, max_len)]
        # timestamps = [(int(self.cam_qr_timesync_result_timestamp) + \
        #                int((self.cam_freq_ms)*1e3)*(cnt-self.cam_qr_timesync_result_frame)) \
        #                     for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        # diff_timestamps = [int((self.cam_freq_ms)*1e3)*(cnt-self.cam_qr_timesync_result_frame)\
        #                     for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        # print(diff_timestamps[:10])
        return frames, timestamps
    
    def estimate_frame_rate(self):
        diff_frame = self.cam_qr_timesync_result_frame_end - self.cam_qr_timesync_result_frame_start
        diff_timestamp = self.cam_qr_timesync_result_timestamp_end - self.cam_qr_timesync_result_timestamp_start
        frame_rate = diff_frame / diff_timestamp
        time_step = 1.0 / frame_rate
        intercept = self.cam_qr_timesync_result_timestamp_start - time_step*self.cam_qr_timesync_result_frame_start
        print("fps:{} ts:{}".format(frame_rate, time_step))
        return frame_rate, time_step, intercept
    
    def interp_cam_timestamps_with_2QRcode(self, max_len):
        fps, ts, intercept = self.estimate_frame_rate()
        
        frames = [cnt for cnt in range(self.cam_qr_timesync_result_frame_start, max_len)]
        # for cnt in range(self.cam_qr_timesync_result_frame_start, max_len):
        #     value = ts*cnt + intercept
        #     print(cnt, value)
        #     time.sleep(1)
        timestamps = [int(ts*cnt + intercept) \
                        for cnt in range(self.cam_qr_timesync_result_frame_start, max_len)]
        # timestamps = [(int(self.cam_qr_timesync_result_timestamp_start) + \
        #                int((ts))*(cnt-self.cam_qr_timesync_result_frame_start)) \
        #                     for cnt in range(self.cam_qr_timesync_result_frame_start, max_len)]
        # diff_timestamps = [int((self.cam_freq_ms)*1e3)*(cnt-self.cam_qr_timesync_result_frame)\
        #                     for cnt in range(self.cam_qr_timesync_result_frame, max_len)]
        # print(diff_timestamps[:10])
        return frames, timestamps
    
    def fine_nearest_timestamp(self, target_timestamp, timestamp_list):
        diff_timestamp = timestamp_list - target_timestamp
        index = np.argmin(np.abs(diff_timestamp))
        # print("nearest_index:{}".format(index))
        return self.cam_qr_timesync_result_frame_start + index
    
    def output_cam_timestamp_interp_result(self, frames, timestamps):
        result = [frames, timestamps]
        result_np = np.array(result).T
        np.savetxt(f'{self.output_dir}{self.interp_timestamp}', result_np, fmt ='%d')
        # np.savetxt(f'./tmp/cam_timestamp_interp_result.csv', result_np, fmt ='%d')
        
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
        image_list = [image_path.replace(f'{self.image_dir}', '').replace('.jpg','') for image_path in image_list]
        # print(image_list)
        # print(self.image_dir)
        if self.cam_qr_timesync_result_frame_end is None:
            frames, interp_timestamps = self.interp_cam_timestamps(int(image_list[-1]))
        else:
            frames, interp_timestamps = self.interp_cam_timestamps_with_2QRcode(int(image_list[-1]))
        self.output_cam_timestamp_interp_result(frames, interp_timestamps)

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

        print('min_frame:{} max_frame:{}'.format(min(self.sim_to_image_index_list), max(self.sim_to_image_index_list)))
    
    
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
    