import time

from tqdm import tqdm
import pickle
import numpy as np
import cv2
import gymnasium as gym
import mediapy as media
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import os
from scenarionet.common_utils import read_dataset_summary, read_scenario
from metadrive.component.sensors.point_cloud_lidar import PointCloudLidar
from metadrive.component.sensors.depth_camera import DepthCamera

class PointCloudLidar_ego_centric(DepthCamera):
    """
    Point cloud lidar in ego-centric coordinate system, with x in front, y in left and z in up
    """
    num_channels = 3  # x, y, z coordinates

    def __init__(self, width, height, ego_centric, engine, *, cuda=False):
        """
        If ego_centric is True, the point cloud will be in the camera's ego coordinate system.
        """
        if cuda:
            raise ValueError("LiDAR does not support CUDA acceleration for now. Ask for support if you need it.")
        super().__init__(width, height, engine, cuda=False)
        self.ego_centric = ego_centric

    def get_rgb_array_cpu(self):
        """
        The result of this function is now a 3D array of point cloud coord in shape (H, W, 3)
        The lens parameters can be changed on the fly!
        """
        lens = self.lens
        fov = lens.getFov()
        f_x = self.BUFFER_W / 2 / (np.tan(fov[0] / 2 / 180 * np.pi))
        f_y = self.BUFFER_H / 2 / (np.tan(fov[1] / 2 / 180 * np.pi))
        intrinsics = np.asarray([[f_x, 0, (self.BUFFER_H - 1) / 2], [0, f_y, (self.BUFFER_W - 1) / 2], [0, 0, 1]])
        f = lens.getFar()
        n = lens.getNear()
        depth = super().get_rgb_array_cpu()
        hpr = self.cam.getHpr()
        # hpr[0] += 90  # pand3d's y is the camera facing direction, so we need to rotate it 90 degree
        #hpr[1] *= -1  # left right handed convert
        rot = R.from_euler('ZYX', hpr, degrees=True)
        rotation_matrix = rot.as_matrix()
        translation = self.cam.getPos()
        translation[0], translation[1], translation[2] = translation[1], -translation[0], translation[2]

        z_eye = 2 * n * f / ((f + n) - (2 * depth - 1) * (f - n))
        points = self.simulate_lidar_from_depth(z_eye.squeeze(-1), intrinsics, translation, rotation_matrix)
        return points

    @staticmethod
    def simulate_lidar_from_depth(depth_img, camera_intrinsics, camera_translation, camera_rotation):
        """
        Simulate LiDAR points in the world coordinate system from a depth image.

        Parameters:
            depth_img (np.ndarray): Depth image of shape (H, W, 3), where the last dimension represents RGB or grayscale depth values.
            camera_intrinsics (np.ndarray): Camera intrinsic matrix of shape (3, 3).
            camera_translation (np.ndarray): Translation vector of the camera in world coordinates of shape (3,).
            camera_rotation (np.ndarray): Rotation matrix of the camera in world coordinates of shape (3, 3).

        Returns:
            np.ndarray: LiDAR points in the world coordinate system of shape (N, 3), where N is the number of valid points.
        """
        # Extract the depth channel (assuming it's grayscale or depth is in the R channel)
        # Get image dimensions
        depth_img = depth_img.T
        depth_img = depth_img[::-1, ::-1]
        height, width = depth_img.shape

        # Create a grid of pixel coordinates (u, v)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        uv_coords = np.stack([u, v, np.ones_like(u)], axis=-1)  # Shape: (H, W, 3)

        # Reshape to (H*W, 3) for easier matrix multiplication
        uv_coords = uv_coords.reshape(-1, 3)

        # Invert the camera intrinsic matrix to project pixels to camera coordinates
        K_inv = np.linalg.inv(camera_intrinsics)

        # Compute 3D points in the camera coordinate system
        cam_coords = (K_inv @ uv_coords.T).T  # Shape: (H*W, 3)
        cam_coords *= depth_img.reshape(-1)[..., None]  # Scale by depth

        # Remove invalid points (e.g., depth = 0)
        # valid_mask = depth_img.flatten() > 0
        # cam_coords = cam_coords[valid_mask]
        cam_coords = cam_coords[..., [2, 1, 0]]

        # Transform points to the world coordinate system
        world_coords = (camera_rotation @ cam_coords.T).T + camera_translation

        # to original shape
        world_coords = world_coords.reshape(height, width, 3)
        return world_coords.swapaxes(0, 1)



def calculate_fov(intrinsic_matrix):
    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]
    w, h = intrinsic_matrix[0, 2] * 2, intrinsic_matrix[1, 2] * 2
    fov_x = 2 * np.arctan(w / (2 * f_x)) * 180 / np.pi
    fov_y = 2 * np.arctan(h / (2 * f_y)) * 180 / np.pi
    return fov_x, fov_y


def angular_sampling(points, h_res=0.2, v_bins=40, v_fov=(-30, 30)):
    def cartesian_to_spherical(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)  # 水平角度
        phi = np.arcsin(z / r)  # 垂直角度
        return r, theta, phi

    # 转换点云为球坐标
    _, theta, phi = cartesian_to_spherical(points)

    # 水平角度离散化
    theta_bins = np.round(theta / np.deg2rad(h_res))

    # 垂直角度离散化
    v_min, v_max = np.deg2rad(v_fov[0]), np.deg2rad(v_fov[1])
    v_res = (v_max - v_min) / v_bins
    phi_bins = ((phi - v_min) / v_res).astype(np.int32)

    # 过滤掉超出范围的点
    valid_mask = (phi_bins >= 0) & (phi_bins < v_bins)
    theta_bins = theta_bins[valid_mask]
    phi_bins = phi_bins[valid_mask]
    points = points[valid_mask]

    # 构建唯一性键值对并筛选
    bins = theta_bins * v_bins + phi_bins  # 将 (theta, phi) 映射为唯一的线性索引
    unique_bins, unique_indices = np.unique(bins, return_index=True)

    return points[unique_indices]

def rotation_matrix_to_euler_angles(rotation_matrix):
    # 创建旋转对象
    r = R.from_matrix(rotation_matrix)
    # 提取欧拉角，使用 ZYX 顺序
    roll, pitch, heading = r.as_euler('xyz', degrees=True)
    return heading, pitch, roll
class CameraAndLidarObservation(BaseObservation):
    def __init__(self, config):
        super(CameraAndLidarObservation, self).__init__(config)
        self.rgb_obs = ImageObservation(config, "rgb_camera", clip_rgb=False)
        self.lidar_obs = ImageObservation(config, "point_cloud", clip_rgb=True)

    @property
    def observation_space(self):
        os = dict(
            rgb=self.rgb_obs.observation_space,
            lidar=self.lidar_obs.observation_space,
        )
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        self.engine.get_sensor("rgb_camera").lens.setFov(fov_x, fov_y)
        self.engine.get_sensor("point_cloud").lens.setFov(fov_x, fov_y)
        ret = {}
        # get rgb camera

        camera_to_world = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        rgb_data = {}
        lidar_data = []
        agent = self.engine.get_sensor("rgb_camera").cam.getParent()

        for k,v in camera_params.items():
            camera_translation = v['sensor2lidar_translation'].copy()
            camera_translation[0], camera_translation[1], camera_translation[2] = -camera_translation[1], camera_translation[0], camera_translation[2]
            camera_rotation = v['sensor2lidar_rotation']@camera_to_world
            h,p,r = rotation_matrix_to_euler_angles(camera_rotation)
            rgb_img = self.rgb_obs.observe(agent, position=camera_translation, hpr=[h,p,r])[..., -1]
            rgb_img = rgb_img[20:1100]
            lidar = self.lidar_obs.observe(agent, position=camera_translation, hpr=[h,p,r])[..., -1]
            lidar = lidar[2:110].reshape(-1,3)
            lidar = lidar[np.linalg.norm(lidar, axis=1) < 100]
            lidar_data.append(lidar)
            rgb_data[k] = rgb_img
        lidar_data = np.concatenate(lidar_data, axis=0)

        ret['camera'] = rgb_data
        ret['lidar'] = lidar_data
        return ret

if __name__ =='__main__':

    data_path = "/Users/fenglan/Dataset/traffic_dataset/processed"
    camera_channel_list = ['CAM_F0', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0', 'CAM_L2', 'CAM_L1', 'CAM_L0']
    rgb_sensor_size = (1920, 1120)
    lidar_sensor_size = (192, 112)

    sample_per_n_frames = 5

    with open('camera_params.pkl', 'rb') as f:
        camera_params = pickle.load(f)

    intrinsics = camera_params['CAM_F0']['intrinsics']
    fov_x, fov_y = calculate_fov(intrinsics)


    summary_dict, summary_list, mapping = read_dataset_summary(data_path)
    num_files = len(summary_list)
    print(f'processing {num_files} scenarios')

    env = ScenarioEnv(
        {
            'agent_observation': CameraAndLidarObservation,
            'image_on_cuda': False,
            # To enable onscreen rendering, set this config to True.
            "use_render": False,

            # !!!!! To enable offscreen rendering, set this config to True !!!!!
            "image_observation": True,
            "norm_pixel": False,
            "stack_size": 1,

            # ===== The scenario and MetaDrive config =====
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "num_scenarios": num_files,
            "horizon": 1000,
            "no_static_vehicles": False,
            "agent_configs": {
                "default_agent": dict(use_special_color=True, vehicle_model="varying_dynamics_bounding_box")
            },
            "vehicle_config": dict(
                show_navi_mark=False,
                show_line_to_dest=False,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50),
            ),
            # "use_bounding_box": True,
            "data_directory": AssetLoader.file_path(data_path, unix_style=False),
            "height_scale": 1,

            "set_static": True,

            # ===== Set some sensor and visualization configs =====
            "daytime": "08:10",
            "window_size": (rgb_sensor_size[0], rgb_sensor_size[1]),
            "camera_dist": 0,  # 0.8, 1.71
            "camera_height": 1.5,  # 1.5
            "camera_pitch": None,
            "sensors": dict(
                point_cloud=(PointCloudLidar_ego_centric, lidar_sensor_size[0],lidar_sensor_size[1], True),
                rgb_camera=(RGBCamera, rgb_sensor_size[0], rgb_sensor_size[1]),

            ),
            # ===== Remove useless items in the images =====
            "show_logo": False,
            "show_fps": False,
            "show_interface": True,
            "disable_collision": True,
            "force_destroy": True,
        }
    )

    for seed in tqdm(range(num_files)):

        rgb_list = []
        lidar_list = []
        scenario_file = summary_list[seed]
        scenario_path = os.path.join(data_path, mapping[scenario_file],scenario_file)
        with open(scenario_path, "rb") as f:
            data = pickle.load(f)
        o, _ = env.reset(seed)
        rgb_list.append(o['camera'])
        lidar_list.append(o['lidar'])

        scenario = env.engine.data_manager.current_scenario
        continue
        horizon = scenario['length']
        rgb_len = horizon // sample_per_n_frames

        for t in tqdm(range(1,horizon)):
            o, r, d, _, _ = env.step([1, 0.88])
            if t % sample_per_n_frames == 0:
                rgb_list.append(o['camera'])
                lidar_list.append(o['lidar'])
        data['synthetic_camera'] = rgb_list
        data['synthetic_lidar'] = lidar_list

        with open(scenario_path, "wb") as f:
            pickle.dump(data, f)
        break

