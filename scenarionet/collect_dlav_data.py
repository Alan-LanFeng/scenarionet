import time
from concurrent.futures import ProcessPoolExecutor

import argparse
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
from metadrive.envs.scenario_env import ScenarioEnv, ScenarioOnlineEnv
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
from numpy import array
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary


camera_params = {'CAM_F0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.00785972, -0.02271912,  0.99971099],
       [-0.99994262,  0.00745516, -0.00769211],
       [-0.00727825, -0.99971409, -0.02277642]]), 'sensor2lidar_translation': array([ 1.65506747, -0.01168732,  1.49112208])}, 'CAM_L0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.81776776, -0.0057693 ,  0.57551942],
       [-0.57553938, -0.01377628,  0.81765802],
       [ 0.0032112 , -0.99988846, -0.01458626]]), 'sensor2lidar_translation': array([1.63069485, 0.11956747, 1.48117884])}, 'CAM_L1': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.93120104,  0.00261563, -0.36449662],
       [ 0.36447127, -0.02048653,  0.93098926],
       [-0.00503215, -0.99978671, -0.0200304 ]]), 'sensor2lidar_translation': array([1.29939471, 0.63819702, 1.36736822])}, 'CAM_L2': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.63520782,  0.01497516, -0.77219607],
       [ 0.77232489, -0.00580669,  0.63520119],
       [ 0.00502834, -0.99987101, -0.01525415]]), 'sensor2lidar_translation': array([-0.49561003,  0.54750373,  1.3472672 ])}, 'CAM_R0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.82454901,  0.01165722,  0.56567043],
       [-0.56528395,  0.02532491, -0.82450755],
       [-0.02393702, -0.9996113 , -0.01429199]]), 'sensor2lidar_translation': array([ 1.61828343, -0.15532203,  1.49007665])}, 'CAM_R1': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.92684778,  0.02177016, -0.37480562],
       [ 0.37497631,  0.00421964, -0.92702479],
       [-0.01859993, -0.9997541 , -0.01207426]]), 'sensor2lidar_translation': array([ 1.27299407, -0.60973112,  1.37217911])}, 'CAM_R2': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[-0.62253245,  0.03706878, -0.78171558],
       [ 0.78163434, -0.02000083, -0.62341618],
       [-0.03874424, -0.99911254, -0.01652307]]), 'sensor2lidar_translation': array([-0.48771615, -0.493167  ,  1.35027683])}, 'CAM_B0': {'distortion': array([-0.356123,  0.172545, -0.00213 ,  0.000464, -0.05231 ]), 'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
       [0.000e+00, 1.545e+03, 5.600e+02],
       [0.000e+00, 0.000e+00, 1.000e+00]]), 'sensor2lidar_rotation': array([[ 0.00802542,  0.01047463, -0.99991293],
       [ 0.99989075, -0.01249671,  0.00789433],
       [-0.01241293, -0.99986705, -0.01057378]]), 'sensor2lidar_translation': array([-0.47463312,  0.02368552,  1.4341838 ])}}


def calculate_fov(intrinsic_matrix):
    f_x = intrinsic_matrix[0, 0]
    f_y = intrinsic_matrix[1, 1]
    w, h = intrinsic_matrix[0, 2] * 2, intrinsic_matrix[1, 2] * 2
    fov_x = 2 * np.arctan(w / (2 * f_x)) * 180 / np.pi
    fov_y = 2 * np.arctan(h / (2 * f_y)) * 180 / np.pi
    return fov_x, fov_y

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
        #self.lidar_obs = ImageObservation(config, "point_cloud", clip_rgb=True)
        self.depth_obs = ImageObservation(config, "depth_camera", clip_rgb=False)

    @property
    def observation_space(self):
        os = dict(
            rgb=self.rgb_obs.observation_space,
            depth=self.depth_obs.observation_space,
        )
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        if self.engine.episode_step!=20: return

        self.engine.get_sensor("rgb_camera").lens.setFov(fov_x, fov_y)
        #self.engine.get_sensor("point_cloud").lens.setFov(fov_x, fov_y)
        ret = {}
        # get rgb camera

        camera_to_world = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        rgb_data = {}
        depth_data = {}
        agent = self.engine.get_sensor("rgb_camera").cam.getParent()

        for k,v in camera_params.items():
            if not k in camera_channel_list: continue
            camera_translation = v['sensor2lidar_translation'].copy()
            camera_translation[0], camera_translation[1], camera_translation[2] = -camera_translation[1], camera_translation[0], camera_translation[2]
            camera_translation[2]+=0.8
            camera_translation[1]-=2
            camera_rotation = v['sensor2lidar_rotation']@camera_to_world
            h,p,r = rotation_matrix_to_euler_angles(camera_rotation)
            p-=3
            rgb_img = self.rgb_obs.observe(agent, position=camera_translation, hpr=[h,p,r])[..., -1]
            rgb_data[k] = rgb_img[:,:,::-1]

            depth = self.depth_obs.observe(agent, position=camera_translation, hpr=[h,p,r])[..., -1]
            depth_data[k] = depth

        # visualize depth and rgb image
        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgb_data['CAM_F0'])
        # plt.title('RGB Image')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(depth_data['CAM_F0'])
        # plt.title('Depth Image')


        ret['camera'] = rgb_data
        ret['depth'] = depth_data
        return ret


def process_data(data):
    camera = data['synthetic_camera']['CAM_F0']
    depth = data['synthetic_depth']['CAM_F0']
    driving_command = data['driving_command'][-1]
    all_tracks = data['tracks']
    sdc_id = data['metadata']['sdc_id']
    sdc_track_state = all_tracks[sdc_id]['state']
    current_index = 20
    sdc_pos = sdc_track_state['position'][:,:2]
    sdc_heading = sdc_track_state['heading']

    # Get reference heading at current index
    ref_heading = sdc_heading[current_index]

    # Compute relative heading (subtract the reference)
    rotated_heading = sdc_heading - ref_heading


    # Rotation matrix to align current heading to x-axis
    cos_h = np.cos(-ref_heading)
    sin_h = np.sin(-ref_heading)
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # Translate all positions so that the current index position is at the origin
    translated_pos = sdc_pos - sdc_pos[current_index]

    # Rotate all positions
    rotated_pos = translated_pos @ rotation_matrix.T  # (n, 2) @ (2, 2).T


    sdc_feature = np.concatenate([rotated_pos, rotated_heading[:,np.newaxis]], axis=-1)
    sdc_history_feature = sdc_feature[:current_index+1]
    sdc_future_feature = sdc_feature[current_index+1:current_index+61]

    return_data = {
        'camera': camera,
        'depth': depth,
        'driving_command': driving_command,
        'sdc_history_feature': sdc_history_feature,
        'sdc_future_feature': sdc_future_feature
    }

    return return_data

def process_scenario(seed):
    """ 处理单个场景文件，适用于多进程运行 """
    print(f"Processing scenario {seed}")
    scenario_file = summary_list[seed]
    scenario_path = os.path.join(data_path, mapping.get(scenario_file, ""), scenario_file)

    # 读取场景数据
    with open(scenario_path, "rb") as f:
        data = pickle.load(f)

    # 创建独立的 env 实例
    env = ScenarioOnlineEnv(
        {
            'render_pipeline': False,
            'agent_observation': CameraAndLidarObservation,
            'image_on_cuda': False,
            "use_render": False,
            "image_observation": True,
            "norm_pixel": False,
            "stack_size": 1,
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "start_scenario_index": 0,
            "num_scenarios": 1,
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
            "data_directory": AssetLoader.file_path(data_path, unix_style=False),
            "height_scale": 1,
            "set_static": True,
            "daytime": "08:10",
            "window_size": (rgb_sensor_size[0], rgb_sensor_size[1]),
            "camera_dist": 0,
            "camera_height": 1.5,
            "camera_pitch": None,
            "sensors": dict(
                depth_camera=(DepthCamera, depth_sensor_size[0], depth_sensor_size[1]),
                rgb_camera=(RGBCamera, rgb_sensor_size[0], rgb_sensor_size[1]),
            ),
            "show_logo": False,
            "show_fps": False,
            "show_interface": True,
            "disable_collision": True,
            "force_destroy": True,
        }
    )
    sd = read_scenario_data(scenario_path, centralize=True)
    env.set_scenario(sd)
    # 复位环境
    o, info = env.reset(0)
    # 存储采样数据
    drving_command = [info['navigation_command']]
    # 获取场景长度
    scenario = env.engine.data_manager.current_scenario
    horizon = scenario['length']
    for t in range(1, horizon):
        o, r, tm, tc, info  = env.step([1, 0.88])
        drving_command.append(info['navigation_command'])
        if t==20:
            break

    data['synthetic_camera'] = o['camera']
    data['synthetic_depth'] = o['depth']
    data['driving_command'] = drving_command

    data = process_data(data)
    with open(f'./dlav_data/{seed}.pkl', "wb") as f:
        pickle.dump(data, f)

    env.close()
    return seed  # 返回已完成的任务索引

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/work/vita/datasets/Scenarionet_Dataset/mini/nuplan")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()
data_path = args.data_path
camera_channel_list = ['CAM_F0']
rgb_sensor_size = (300, 200)
depth_sensor_size = (300, 200)
sample_per_n_frames = 5

intrinsics = camera_params['CAM_F0']['intrinsics']
fov_x, fov_y = calculate_fov(intrinsics)

summary_dict, summary_list, mapping = read_dataset_summary(data_path)
num_files = len(summary_list)
print(f'processing {num_files} scenarios')


if __name__ == '__main__':
    os.makedirs('./dlav_data', exist_ok=True)
    try:
        from mpi4py import MPI
        from tqdm import tqdm
        import time
        # 初始化 MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()  # 当前进程的 ID
        device_id = rank
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
        print(f"Process {rank}: CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        size = comm.Get_size()  # 总进程数
        # 计算每个进程需要处理的文件索引
        files_per_rank = num_files // size
        extra = num_files % size  # 处理不能整除的情况
        if rank < extra:
            start_idx = rank * (files_per_rank + 1)
            end_idx = start_idx + files_per_rank + 1
        else:
            start_idx = rank * files_per_rank + extra
            end_idx = start_idx + files_per_rank

        assigned_files = list(range(start_idx, end_idx))

        # 处理任务
        results = [process_scenario(f) for f in tqdm(assigned_files, desc=f"Process {rank}")]

        # 进程 0 收集所有结果
        all_results = comm.gather(results, root=0)

        # 仅在 rank 0 上显示最终结果
        if rank == 0:
            all_results = [item for sublist in all_results for item in sublist]  # 展平列表
            print("\nFinal Results:")
            for r in all_results:
                print(r)
    except:
        print("MPI not available, using ProcessPoolExecutor")
        #process_scenario(0)
        os.makedirs('./dlav_data', exist_ok=True)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(process_scenario, range(num_files)), total=num_files))