import numpy as np
import yaml
from camera import Camera
from groundtruth import Groundtruth
from feature import ShiTomasiDetector
from visual_odometry import VisualOdometry
from dataset import VideoDataset
from mplot import Mplot3d

camera_settings_file = "data\kitti06\KITTI04-12.yaml"
groundtruth_file = "data\kitti06\groundtruth.txt"
dataset_file = "data/kitti06/video.mp4"
if __name__ == "__main__":
    
    with open(camera_settings_file, 'r') as stream:
        cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
    cam = Camera(cam_settings)
    groundtruth = Groundtruth(groundtruth_file) 
    feature_tracker = ShiTomasiDetector()
    vo = VisualOdometry(cam, groundtruth, feature_tracker)
    dataset = VideoDataset(dataset_file)
    
    plt3d = Mplot3d(title='3D trajectory')

    img_id = 0
    while (img_id < dataset.num_frames):

        img = dataset.getImage(img_id)

        if img is not None:

            vo.track(img, img_id)  # main VO function 

            if(img_id > 2):	       
                plt3d.drawTraj(vo.traj3d_gt,'ground truth',color='r',marker='.')
                plt3d.drawTraj(vo.traj3d_est,'estimated',color='g',marker='.')
        img_id += 1
    plt3d.quit()

