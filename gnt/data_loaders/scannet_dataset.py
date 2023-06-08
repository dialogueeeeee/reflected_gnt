import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import glob
import sys

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from .utils.base_utils import downsample_gaussian_blur
from .asset import *


# only for training
class RendererDataset(Dataset):
    def __init__(self, args, is_train, **kwargs):
        self.is_train = is_train
        if self.is_train == 'train':
            self.scene_path_list = scannet_train_scans_320
        else:
            self.scene_path_list = scannet_val_scans_320

        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        image_size = 320
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        all_rgb_files = []
        all_pose_files = []
        all_intrinsics_files = []
        for i, scene_path in enumerate(self.scene_path_list):
            scene_path = os.path.join(args.rootdir + 'data', scene_path[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose = np.loadtxt(path)
                # pose = self.pose_inverse(np.loadtxt(path).reshape(4, 4)[:3, :])
                
                if np.isinf(pose).any() or np.isnan(pose).any():
                    continue
                else:
                    pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            intrinsics_files = [
                os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
            ]

            all_rgb_files.append(rgb_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def pose_inverse(self, pose):
        R = pose[:, :3].T
        t = - R @ pose[:, 3:]
        inversed_pose = np.concatenate([R, t], -1)
        return np.concatenate([inversed_pose, [[0, 0, 0, 1]]])
        # return inversed_pose
    
    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        id_render = np.random.choice(np.arange(len(pose_files)))
        # train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        train_poses = np.stack([self.pose_inverse(np.loadtxt(file).reshape(4, 4)[:3, :]) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]

        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views * subsample_factor,
            tar_id=id_render,
            angular_dist_method="vector",
        )
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        if id_render in id_feat:
            assert id_render not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = id_render

        rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.0

        if self.w != 1296:
            rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
        intrinsics = np.loadtxt(intrinsics_files[id_render])
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )

        all_poses = [render_pose]
        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        depth_range = torch.tensor([near_depth, far_depth])
        # depth_range = torch.tensor([0.1, 10.0])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
            if self.w != 1296:
                src_rgb = cv2.resize(downsample_gaussian_blur(
                    src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            # pose = np.loadtxt()
            pose = self.pose_inverse(np.loadtxt(pose_files[id]).reshape(4, 4)[:3, :])

            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id])
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                np.float32
            )
            src_cameras.append(src_camera)
            all_poses.append(pose)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {
            "rgb": torch.from_numpy(rgb),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_files[id_render],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
