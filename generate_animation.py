import os
import ffmpeg
import numpy as np
import subprocess
from PIL import Image
from tqdm import tqdm

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
        self.processed_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/mixed_images'
        
        os.makedirs(self.processed_image_path, exist_ok=True)
        
        self.pose_image_root_dir = f'/home/kenta/ego_exo/main/01_walk/002_walk/processed_data/vis_contact_poses3d/'
        self.target_cam = '/cam01/rgb/'
        self.pose_image_path = f'{self.pose_image_root_dir}{self.target_cam}'
        
        self.output_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}'
        self.output_video_name = f'{self.output_path}/processed/processed_video.mp4'
        self.frame_rate = 20
            
            
    def generate_mp4(self, timesync_info):
        pose_init_frame = int(timesync_info[0,1]) - 1
        start_frame = 30
        
        # for frame_list in tqdm(timesync_info, desc='Generating combined images', unit='frames'):
        for frame_list in tqdm(timesync_info, desc='Generating combined images', unit='frames'):
            if (int(frame_list[0])+start_frame) < len(timesync_info) - 1:
                # === Generate mp4 data ===
                # print("SimFrame:{} | PoseFrame:{}".format((int(frame_list[0])+start_frame), (int(frame_list[1])-pose_init_frame)))
                # print(f'{self.sim_image_path}{(int(frame_list[0])+start_frame):05d}.png')
                image1_path = f'{self.sim_image_path}{(int(frame_list[0])+start_frame):05d}.png'  # Simulation image
                image2_path = f'{self.pose_image_path}{(int(frame_list[1])-pose_init_frame):05d}.jpg'  # Pose image
                image3_path = f'{self.pose3d_image_path}{(int(frame_list[1]-pose_init_frame)):05d}.png'  # Pose3d image
                image_paths = [image1_path, image2_path, image3_path]
                
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
                result_image.save(frame_filename)
                
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