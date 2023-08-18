import cv2
import numpy as np
import os
import traceback
from tqdm import tqdm

freq_sec = 1/30
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
    try:
        detect = cv2.QRCodeDetector()
        value, points, straight_qrcode = detect.detectAndDecode(img)
        
        if value:
            print("QR Code detected - Frame:{} Value:{}".format(idx, value))
            if (timestamp == value):
                diff_frame = idx - frame_id
                value = interp_timestamp(timestamp, diff_frame)
                print("[SameQR] Interp_time:{} Diff_frame:{} Diff_time:{}".format(float(value), int(diff_frame), value - float(timestamp)))      
            else:
                diff_frame = idx - frame_id
                est_time = interp_timestamp(timestamp, diff_frame)
                print("[RefreshQR] CurrTime:{} InterpTime:{} Diff:{}".format(value, est_time, est_time-float(value)))
                timestamp = value
                frame_id = idx

            # Draw a rectangle around the QR code
            if len(points) > 0:
                num_points = len(points[0])
    
            # # Display the image in a window
            # window_name = f'FrameID:{idx}'
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(window_name, 600, 400) 
            # cv2.imshow(window_name, img)
            # # Wait for a key press and then close the window
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()  
        # else:
            # print("No QR Code detected.")
        
        return int(value)
        # else:
        #     return -1
        
    ## print the exceptiona
    except Exception as e:
        # traceback.print_exc()
        print(e.args[0])
        return 0

# just for cam1
def read_qr_code_double(img):
    """
    get an image and read the QR code.
    Erwin's time stamp are all digits with a length of 13
    Kenta's time stamp with period and was 17-digit long
    """    
    ts1=0 #kenta's timestamp
    ts2=0 #erwin's timestamp
    try:
        detect = cv2.QRCodeDetector()
        results = detect.detectAndDecodeMulti(img)
        
        # # Display the image in a window
        # window_name = f'FrameID:{idx}'
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, 600, 400) 
        # cv2.imshow(window_name, img)
        # # Wait for a key press and then close the window
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()  
        
        
        for value in results[1]:
            #this is a hard code to check kenta's qr code
            if "." in value:
                #remove the period and trim to 13 digit
                ts1 = value.replace('.', '')[:13]
                
                ts1 = value.replace('.', '')
            #this is erwin's code
            elif len(value)==13:
                ts2 = value
        return int(ts1), int(ts2)
    except:
        return 0, 0



# ##--------read aria qr codes--------------------------------------------
data_dir = '/media/kenta/ExtremePro/dataset/ego_exo/230817'
time_synced_data_dir = '/media/kenta/ExtremePro/ego_exo/common/time_synced_exo'
sequence_name = '01_walk'

# ##--------read cam01--------------------------------------------
# cam = 'aria01'; start_idx = 8060; end_idx = 8240
cam = 'cam01'; start_idx = 3830; end_idx = 3960
# cam = 'cam02'; start_idx = 60; end_idx = 405
# cam = 'cam03'; start_idx = 10; end_idx = 210
# cam = 'cam04'; start_idx = 30; end_idx = 260
# cam = 'cam05'; start_idx = 30; end_idx = 80
# cam = 'cam06'; start_idx = 40; end_idx = 160
# cam = 'cam07'; start_idx = 1; end_idx = 150
# cam = 'cam08'; start_idx = 40; end_idx = 140

## For Simulator to GoPro timesync
if READ_SIM_QR:
    # cam = 'cam01'; start_idx = 3830; end_idx = 3960 
    cam = 'cam01'; start_idx = 24990; end_idx = 25330 
    

##----------------------------------------------------------------
if cam.startswith('cam'):
    rgb_dir = os.path.join(time_synced_data_dir, sequence_name, 'exo', cam, 'images')

else:
    rgb_dir = os.path.join(data_dir, sequence_name, cam, 'images', 'rgb')
    
# Debug:
# rgb_dir = os.path.join(data_dir, sequence_name, cam, 'vrs_images', '214-1')

###---------------------------------------------------------------------
rgb_images = sorted(os.listdir(rgb_dir))

valid_rgb_images = []
valid_ts = []


for idx, rgb_image in enumerate(tqdm(rgb_images)):
    
    if idx < start_idx or idx > end_idx:
        continue

    image = cv2.imread(os.path.join(rgb_dir, rgb_image))

    if cam == 'cam01':
        if READ_SIM_QR:
            ts, _ = read_qr_code_double(image); print('double qr')
        else:
            ts = read_qr_code_single(idx, image); 
    else:
        ts = read_qr_code_single(idx, image)
    
    if ts != 0:
        valid_rgb_images.append(rgb_image)
        
        if ts != -1:
            # ts = ts - 1675291000000
            ts = ts
            valid_ts.append(ts)

        print(ts, rgb_image)
 
        


print('------all timestamps------')
print(sequence_name, cam)
for ts, rgb_image in zip(valid_ts, valid_rgb_images):
    print('{} {}'.format(str(ts).zfill(8), rgb_image))