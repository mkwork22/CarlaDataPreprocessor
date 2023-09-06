import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import settings as sets

SKELETON = {
                'left_leg': [13, 15], ## l-knee to l-ankle
                'right_leg': [14, 16], ## r-knee to r-ankle
                'left_thigh': [11, 13], ## l-hip to l-knee
                'right_thigh': [12, 14], ## r-hip to r-knee
                'hip': [11, 12], ## l-hip to r-hip
                'left_torso': [5, 11], ## l-shldr to l-hip
                'right_torso': [6, 12], ## r-shldr to r-hip
                'left_bicep': [5, 7], ## l-shldr to l-elbow
                'right_bicep': [6, 8], ## r-shldr to r-elbow
                'shoulder': [5, 6], ## l-shldr to r-shldr
                'left_hand': [7, 9], ## l-elbow to l-wrist
                'right_hand': [8, 10], ## r-elbow to r-wrist
                'left_face': [1, 0], ## l-eye to nose
                'right_face': [2, 0], ## l-eye to nose
                'face': [1, 2], ## l-eye to r-eye
                'left_ear': [1, 3], ## l-eye to l-ear
                'right_ear': [2, 4], ## l-eye to r-ear
                'left_neck': [3, 5], ## l-ear to l-shldr
                'right_neck': [4, 6], ## r-ear to r-shldr
}

def get_target_file_list(dir, ext):
    target_dir = f'{dir}/**/*.{ext}'
    print(target_dir)
    fpath_list = sorted(glob.glob(f'{dir}/**/*.{ext}', recursive=True))
    return fpath_list


class Visualzier_():
    def __init__(self):
        # self.target_dir = f'/home/kenta/ego_exo/main/01_walk/002_walk/processed_data/contact_poses3d'
        self.target_dir = f'{sets.POSE_DATA_ROOT_DIR}{sets.BIG_SEQUENCE}{sets.SUB_SEQUENCE}processed_data/contact_poses3d'
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.simdata_root_dir = sets.SIMDATA_ROOT_DIR
        self.logdate = sets.LOGDATE
        self.logdir = sets.LOGDIR
        self.processed_image_path = f'{self.simdata_root_dir}{self.logdate}{self.logdir}processed/pose3d'
        os.makedirs(self.processed_image_path, exist_ok=True)
        
        self.skelton = SKELETON

    def load_npy(self, target_fname):
        # target = f'{self.target_dir}/{fname}'
        # print(target_fname)
        return np.load(target_fname, allow_pickle=True).item()['aria01']

    
    def visualize(self, iter, pose3d):
        # print(pose3d)
       
        # Visualize keypoints
        self.ax.scatter(pose3d[:, 0], pose3d[:, 2], -pose3d[:, 1], c='r', marker='o')
        
        # Visualize link
        for connection in self.skelton.values():
            start_index, end_index = connection
            x = pose3d[[start_index, end_index], 0]
            y = pose3d[[start_index, end_index], 1]
            z = pose3d[[start_index, end_index], 2]
            self.ax.plot(x,z,-y, c='b')

        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        self.ax.set_xlim(np.min(pose3d[:, 0]), np.max(pose3d[:, 0]))
        self.ax.set_ylim(np.min(pose3d[:, 2]), np.max(pose3d[:, 2]))
        self.ax.set_zlim(np.min(-pose3d[:, 1]), np.max(-pose3d[:, 1]))
        self.ax.set_aspect('equal')
        
        plt.gca().axis('off')
        # plt.pause(1/30)
        
        plt.savefig(f'{self.processed_image_path}/{iter:05d}.png', format="png", dpi=300, bbox_inches='tight', pad_inches=0)
        self.ax.clear()
                
    def execute(self):
        fpath_list = get_target_file_list(self.target_dir, 'npy')
        filelist = [dir.replace(f'{self.target_dir}/', '') for dir in fpath_list]
        # print(filelist)
        iter = 0
        
        for file in tqdm(fpath_list, desc='Generating pose images', unit='frames'):
            iter+=1
            pose3d = self.load_npy(file)
            self.visualize(iter, pose3d)
            
        
def main():
    visualizer = Visualzier_()
    visualizer.execute()
    
if __name__ == '__main__':
    main()
        