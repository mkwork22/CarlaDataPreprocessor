import os
import glob
import traceback
from datetime import datetime

def create_path(dirname):
    try:
        os.makedirs(dirname, exist_ok=True)
    except:
        print("[WARN] Error to create directory:" + dirname)
        traceback.print_exc()
        pass
    
def get_date():
    now = datetime.now()
    dt_string = now.strftime("%m%d%Y")
    # print("date and time =", dt_string)
    return dt_string
    
def get_time():
    now = datetime.now()
    dt_string = now.strftime("%H%M%S")
    # print("date and time =", dt_string)	
    return dt_string

def get_csv_file_list(dir):
    fpath_list = sorted(glob.glob(dir + "/**/**/*.csv", recursive=True))
    fpath_list.extend(sorted(glob.glob(dir + "/**/*-frames_first*.txt", recursive=True)))
    return fpath_list

def get_log_csv_file_list(dir):
    pattern = os.path.join(dir, '**', 'logdata*.csv')
    fpath_list = glob.glob(pattern, recursive=True)
    return fpath_list

def get_jpg_file_list(dir):
    fpath_list = sorted(glob.glob(dir + "/**/*.jpg", recursive=True))
    return fpath_list

def get_target_file_list(dir, ext):
    target_dir = f'{dir}/**/*.{ext}'
    fpath_list = sorted(glob.glob(f'{dir}/**/*.{ext}', recursive=True))
    return fpath_list
