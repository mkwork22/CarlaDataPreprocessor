import os
import ffmpeg
import numpy as np
import subprocess
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from data_manager import DataManager_
import utility as utils
import settings as sets

class AnimationGenerator_():
    def __init__(self):
        self.simdata_root_dir = sets.SIMDATA_ROOT_DIR
        self.logdate = sets.LOGDATE
        self.logdir = sets.LOGDIR
        self.sim_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}image/'
        self.pose3d_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/pose3d/'
        self.carla_1st_person_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/1stperson/images/'
        self.carla_3rd_person_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/3rdperson/images/'
        self.processed_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/mixed_images'
        # self.processed_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/mixed_images2'

        os.makedirs(self.processed_image_path, exist_ok=True)
        
        # self.pose_image_root_dir = f'/media/kenta/ExtremePro/ego_exo/main/01_walk/001_walk/processed_data/vis_contact_poses3d/'
        self.pose_image_root_dir = f'{sets.POSE_DATA_ROOT_DIR}{sets.BIG_SEQUENCE}{sets.SUB_SEQUENCE}'
        # self.target_cam = '/cam01/rgb/'
        # self.pose_image_path = f'{self.pose_image_root_dir}{self.target_cam}'
        self.pose_image_path = f'{self.pose_image_root_dir}{sets.POSE_SUB_DIR}'
        

        self.output_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}'
        self.output_video_name = f'{self.output_path}/processed/processed_video_comp.mp4'
        # self.output_video_name = f'{self.output_path}/processed/processed_video_with_Carla_image_3rd.mp4'
        self.frame_rate = 20

    def generate_frame(self, frame_list, start_frame, pose_init_frame, timesync_info):
        # frame_list = frame_info[0]
        if (int(frame_list[0])+start_frame) < len(timesync_info) - 1:
        # if (int(frame_list[0])+start_frame) < 300:
            # === Generate mp4 data ===
            print("SimFrame:{} | PoseFrame:{}".format((int(frame_list[0])+start_frame), (int(frame_list[1])-pose_init_frame)))
            # print(f'{self.sim_image_path}{(int(frame_list[0])+start_frame):05d}.png')
            image0_path = f'{self.pose_image_path}{(int(frame_list[1])-pose_init_frame):05d}.jpg' # Basic GoPro image
            image1_path = f'{self.sim_image_path}{(int(frame_list[0])+start_frame):05d}.png'  # Simulation BEV image
            image2_path = f'{self.pose_image_path}{(int(frame_list[1])-pose_init_frame):05d}.jpg'  # Pose image
            image3_path = f'{self.pose3d_image_path}{(int(frame_list[1]-pose_init_frame)):05d}.png'  # Pose3d image
            image4_path = f'{self.carla_1st_person_image_path}{(int(frame_list[0])+start_frame):05d}.png'
            image5_path = f'{self.carla_3rd_person_image_path}{(int(frame_list[0])+start_frame):05d}.png'
            image_paths = [image1_path, image2_path, image3_path]
            # image_paths = [image5_path, image2_path, image3_path]
            # image_paths = [image5_path, image2_path, image3_path]
            # image_paths = [image0_path, image1_path]

            images = [Image.open(path) for path in image_paths]
            widths, heights = zip(*(i.size for i in images))
            
            # Fit aspect ratio
            max_height = max(heights)
            resized_images = [image.resize((int(width * max_height / height), max_height)) for image, width, height in zip(images, widths, heights)]

            # Resize image
            total_width = sum([image.width for image in resized_images])
            result_image = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for image in resized_images:
                result_image.paste(image, (x_offset, 0))
                x_offset += image.width
                
            # Save frame image
            frame_filename = os.path.join(self.processed_image_path, f'{int(frame_list[0]):05d}.png')
            # print(frame_filename)
            result_image.save(frame_filename)
            
    def generate_mp4(self, timesync_info):
        pose_init_frame = int(timesync_info[0,1]) - 1
        start_frame = sets.CONST_SYS_DELAY # compensate transmittor delay (wireless transmission)

        # for frame_list in tqdm(timesync_info, desc='Generating combined images', unit='frames'):
        #     self.generate_frame(frame_list, start_frame, pose_init_frame, timesync_info)
        
        with ProcessPoolExecutor(max_workers=20) as executor:
            for frame_list in tqdm(timesync_info, desc='Generating combined images', unit='frames'):
                executor.submit(self.generate_frame, frame_list, start_frame, pose_init_frame, timesync_info)

        # Generate video
        subprocess.run(['ffmpeg', '-r', str(self.frame_rate), '-f', 'image2', '-i', os.path.join(self.processed_image_path, '%05d.png'), '-pix_fmt', 'yuv420p', self.output_video_name])
        
    def execute(self):
        # === Load timesync info ===
        timesync_info_dir = f'{self.simdata_root_dir}{self.logdate}{self.logdir}'
        timesync_info = np.loadtxt(f'{timesync_info_dir}timesync_result.csv')
        
        # === Load simdata ===
        sim_file_list = utils.get_target_file_list(self.sim_image_path, 'png')
        
        # === Load pose image file list ===
        pose_file_list = utils.get_target_file_list(self.pose_image_path, 'jpg')
        
        # === Create image path sets from timesync info ===
        # === Create MP4 video ===
        self.generate_mp4(timesync_info)

def main():    
    anim_generator = AnimationGenerator_()
    anim_generator.execute()

if __name__ == '__main__':
    main()