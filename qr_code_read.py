import cv2
import numpy as np
import os
import traceback
from tqdm import tqdm

import settings as sets

freq_sec = 1/31
global timestamp
global frame_id
# Init variables
timestamp = int(0)
frame_id = 0
READ_SIM_QR = True

def interp_timestamp(curr_time, cnt):
    return int(curr_time) + int(freq_sec*1e3) * cnt
    
# for camera other than the calibration one
def read_qr_code_single(idx, img):
    """get an image and read the QR code.
    """    
    global timestamp
    global frame_id
    detect_refresh_qr = False

    try:
        detect = cv2.QRCodeDetector()
        value, points, straight_qrcode = detect.detectAndDecode(img)
        
        if value:
            # print("QR Code detected - Frame:{} Value:{}".format(idx, value))
            if (timestamp == value):
                diff_frame = idx - frame_id
                value = interp_timestamp(timestamp, diff_frame)
                # print("[SameQR] Interp_time:{} Diff_frame:{} Diff_time:{}".format(float(value), int(diff_frame), value - float(timestamp)))      
            else:
                diff_frame = idx - frame_id
                est_time = interp_timestamp(timestamp, diff_frame)
                # print("[RefreshQR] CurrTime:{} InterpTime:{} Diff:{}".format(value, est_time, est_time-float(value)))
                timestamp = value
                frame_id = idx
                detect_refresh_qr = True

            # Draw a rectangle around the QR code
            if len(points) > 0:
                num_points = len(points[0])
    
        #     # # Display the image in a window
        #     # window_name = f'FrameID:{idx}'
        #     # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        #     # cv2.resizeWindow(window_name, 600, 400) 
        #     # cv2.imshow(window_name, img)
        #     # # Wait for a key press and then close the window
        #     # cv2.waitKey(1000)
        #     # cv2.destroyAllWindows()  
        # # else:
        #     # print("No QR Code detected.")
        
            return int(value), frame_id, detect_refresh_qr
        else:
            return 0, None, None
        
    ## print the exceptiona
    except Exception as e:
        traceback.print_exc()
        # print(e.args[0])
        return 0, None, None

# just for cam1
def read_qr_code_double(idx, img):
    """
    get an image and read the QR code.
    Erwin's time stamp are all digits with a length of 13
    Kenta's time stamp with period and was 17-digit long
    """    
    global timestamp
    global frame_id
    ts1=0 #kenta's timestamp
    ts2=0 #erwin's timestamp
    detect_refresh_qr = False
    try:
        detect = cv2.QRCodeDetector()
        results = detect.detectAndDecodeMulti(img)
        
        # # Display the image in a window
        # window_name = f'FrameID:{idx}'
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, 600, 400) 
        # cv2.imshow(window_name, img)
        # # Wait for a key press and then close the window
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()  
        
        
        for value in results[1]:
            #this is a hard code to check kenta's qr code
            if "." in value:
                #remove the period and trim to 13 digit
                # ts1 = value.replace('.', '')[:13]
                ts1 = value.replace('.', '')

                # print("QR Code detected - Frame:{} Value:{}".format(idx, ts1))
                if (timestamp == ts1):
                    diff_frame = idx - frame_id
                    # value = interp_timestamp(timestamp, diff_frame)
                    # print("[SameQR] Interp_time:{} Diff_frame:{} Diff_time:{}".format(float(ts1), int(diff_frame), ts1 - float(timestamp)))      
                else:
                    diff_frame = idx - frame_id
                    est_time = interp_timestamp(timestamp, diff_frame)
                    # print("[RefreshQR] CurrTime:{} InterpTime:{} Diff:{}".format(ts1, est_time, est_time-float(value)))
                    timestamp = ts1
                    frame_id = idx
                    detect_refresh_qr = True

            #this is erwin's code
            elif len(value)==13:
                ts2 = value
        return int(ts1[:13]), frame_id, detect_refresh_qr
    except:
        # traceback.print_exc()
        return 0, 0, False



# ##--------read aria qr codes--------------------------------------------
# data_dir = '/home/kenta/ego_exo/common/raw_from_cameras'
# time_synced_data_dir = '/home/kenta/ego_exo/common/time_synced_exo'
data_dir = '/media/kenta/ExtremePro/dataset/ego_exo/230817'
time_synced_data_dir = '/media/kenta/ExtremePro/ego_exo/common/time_synced_exo'
# sequence_name = '03_walk'
sequence_name = sets.BIG_SEQUENCE

# ##--------read cam01--------------------------------------------
cam = 'cam01'; start_idx = 1990; end_idx = 2080

# cam_list = ['cam01', 'cam02', 'cam03', 'cam04', 'cam05', 'cam06', 'cam07', 'cam08']
cam_list = ['cam01', 'cam01']
# start_idx_list = [3800, 25200] # celine
# start_idx_list = [7430, 23560] # nathaniel
# start_idx_list = [3940, 21400] # darren
start_idx_list = [4420, 20840] # dan


## For Simulator to GoPro timesync
if READ_SIM_QR:
    cam = 'cam01'; end_idx = 30000 

extracted_infos = []

for cam, start_idx in zip(cam_list, start_idx_list):
    print("processing {}".format(cam))
    valid_rgb_images = []
    valid_ts = []
    extracted_timestamps = []
    extracted_frames = []

    print(start_idx)

    ##----------------------------------------------------------------
    if cam.startswith('cam'):
        rgb_dir = os.path.join(time_synced_data_dir, sequence_name, 'exo', cam, 'images')

    else:
        rgb_dir = os.path.join(data_dir, sequence_name, cam, 'images', 'rgb')
        
    # Debug:
    # rgb_dir = os.path.join(data_dir, sequence_name, cam, 'vrs_images', '214-1')

    ###---------------------------------------------------------------------
    rgb_images = sorted(os.listdir(rgb_dir))

    for idx, rgb_image in enumerate(tqdm(rgb_images)):
        if idx < start_idx or idx > end_idx:
            continue
            
        # print(f'frame:{idx}')
            
        image = cv2.imread(os.path.join(rgb_dir, rgb_image))

        if cam == 'cam01':
            if READ_SIM_QR:
                ts, frame, detect_refresh_qr = read_qr_code_double(idx, image)
            else:
                ts, frame, detect_refresh_qr = read_qr_code_single(idx, image) 
        else:
            ts, frame, detect_refresh_qr = read_qr_code_single(idx, image)
        
        if ts != 0:
            valid_rgb_images.append(rgb_image)

            if ts != -1:
                if (READ_SIM_QR):
                    ts = ts
                else:
                    ts = ts - 1675291000000
                # ts = ts
                valid_ts.append(ts)

            if detect_refresh_qr:
                extracted_timestamps.append(f'{frame:05d}:{ts:013d}')
                print(f'{frame:05d}:{ts:013d}')
                extracted_frames.append(frame)
            # print(extracted_timestamps)
            # print(ts, rgb_image)
                if len(extracted_timestamps) > 3:
                    print("break")
                    break

    print(f'{cam}:{extracted_timestamps}')
    print(extracted_timestamps[1])
    # extracted_infos.append(f'{cam}:{extracted_timestamps[1]}')
    extracted_infos.append(f'{cam}:{extracted_timestamps[1]}')
    # start_idx = extracted_frames[-1]
    # end_idx = extracted_frames[-1] + 1000
    
print('------all timestamps------')
for info in extracted_infos:
    print(f'{info}--', end='')


# print(sequence_name, cam)
# for ts, rgb_image in zip(valid_ts, valid_rgb_images):
#     print('{} {}'.format(str(ts).zfill(8), rgb_image))
