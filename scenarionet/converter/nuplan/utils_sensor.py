"""
Adapted from https://github.com/valeoai/VisualQuantization/blob/dev/scripts/nuplan_preprocessing/gen_nuplan_pickle.py

This script extract perception info from a NuPlan scenario in one list of python dicts with the following form:

```angular2html
{
    'scene': {
        name: <string> --- identifier of the scene the sample,
        description: <string> --- e.g., 'rain'
        timestamp:<int> --- using 'CAM_F0' timestep as reference timestamp for indexing
    }

    'ego_state':{
        ego_center: <float> [3] -- Ego vehicle location in global coordinates in meters: x, y, z.
        ego_heading: ego orientation as quaternion: w, x, y, z
        ego_velocity: <float> [3] -- Ego vehicle velocity in global coordinates in m/s: vx, vy, vz=0
        ego_acceleration: <float> [3] -- Ego vehicle acceleration in global coordinates in m/s**2: ax, ay, az=0
    }

    'CAM_F0': { # CAM_FRONT
        'file_path': path to image file from dataroot,
        'timestamp': <int> -- Unix time stamp at which the data (image and ego pose) have been recorded for this sensor,
        'intrinsic': <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras,
        'distortion':<float> [5] -- camera_distortion in Caltech model (k1, k2, p1, p2, k3),
        'sensor_to_ego_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z,
        'sensor_to_ego_tran': <float> [3] -- Coordinate system origin in meters: x, y, z,
        'ego_to_world_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z,
        'ego_to_world_tran': <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0,
    },
    'CAM_L0': {...}, # CAM_FRONT_LEFT
    'CAM_R0': {...}, # CAM_FRONT_RIGHT
    'CAM_L1': {...}, # CAM_MID_LEFT
    'CAM_R1': {...}, # CAM_MID_RIGHT
    'CAM_L2': {...}, # CAM_BACK_LEFT
    'CAM_R2': {...}, # CAM_BACK_RIGHT
    'CAM_B0': {...}, # CAM_BACK
    'LIDAR_TOP': {
        'file_path': path to pointcloud file from dataroot,
        'timestamp': <int> -- Unix time stamp at which the data (pointcloud and ego pose) have been recorded for this sensor,
        'sensor_to_ego_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z,
        'sensor_to_ego_tran': <float> [3] -- Coordinate system origin in meters: x, y, z,
        'ego_to_world_rot': <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z,
        'ego_to_world_tran': <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0,
    },
}
```
"""
import os
import pickle
import sqlite3

from typing import Any, Dict, List, Tuple
from contextlib import contextmanager
from collections import defaultdict

import numpy as np


@contextmanager
def get_db_cursor(db_file: str):
    """
    Context manager for database cursors.

    Args:
        db_file: Path to the database file.

    Yields:
        A sqlite3 cursor object.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        conn.close()


def fetch_lidar_data_scenario(cursor: sqlite3.Cursor, lidar_tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches LiDAR data for a given list of lidar tokens.
    """
    query = f"""
    SELECT lp.token, lp.timestamp, lp.ego_pose_token, lp.filename, l.channel, l.translation, l.rotation
    FROM lidar_pc lp
    JOIN lidar l ON lp.lidar_token = l.token
    WHERE lp.token IN ({','.join(['?'] * len(lidar_tokens))})
    ORDER BY lp.timestamp
    """
    cursor.execute(query, [bytes.fromhex(t) for t in lidar_tokens])
    rows = cursor.fetchall()

    lidar_data = []
    for row in rows:
        token, timestamp, ego_pose_token, filename, channel, translation, rotation = row
        translation = pickle.loads(translation)
        rotation = pickle.loads(rotation)

        lidar_data.append({
            'token': token.hex(),
            'timestamp': timestamp,
            'ego_pose_token': ego_pose_token.hex(),
            'filename': filename,
            'channel': channel,
            'sensor_to_ego_translation': list(translation),
            'sensor_to_ego_rotation': list(rotation),
        })
    return lidar_data


def fetch_ego_poses_scenario(cursor: sqlite3.Cursor, lidar_tokens: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches ego pose data only for the ego_pose_token associated with the given lidar_tokens.

    Args:
        cursor: SQLite cursor.
        lidar_tokens: List of lidar tokens for the scenario.

    Returns:
        Dictionary of ego poses keyed by token.
    """
    if not lidar_tokens:
        return {}

    query = f"""
    SELECT ep.token, ep.timestamp, ep.x, ep.y, ep.z, ep.qw, ep.qx, ep.qy, ep.qz,
           ep.vx, ep.vy, ep.vz, ep.acceleration_x, ep.acceleration_y, ep.acceleration_z
    FROM ego_pose ep
    JOIN lidar_pc lp ON ep.token = lp.ego_pose_token
    WHERE lp.token IN ({','.join(['?'] * len(lidar_tokens))})
    """

    cursor.execute(query, [bytearray.fromhex(t) for t in lidar_tokens])
    rows = cursor.fetchall()

    ego_poses = {}
    for row in rows:
        token = row[0].hex()
        ego_poses[token] = {
            'timestamp': row[1],
            'position': [row[2], row[3], row[4]],
            'orientation': [row[5], row[6], row[7], row[8]],
            'velocity': [row[9], row[10], row[11]],
            'acceleration': [row[12], row[13], row[14]]
        }

    return ego_poses


def fetch_camera_data_scenario(cursor: sqlite3.Cursor, lidar_tokens: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches camera data for the given lidar timestamps.

    Args:
        cursor: SQLite cursor.
        lidar_tokens: List of lidar tokens for the scenario.

    Returns:
        Dictionary of camera data per channel.
    """
    if not lidar_tokens:
        return {}

    query = f"""
    SELECT c.channel, i.token, i.filename_jpg, i.timestamp, i.ego_pose_token,
           c.translation, c.rotation, c.intrinsic, c.distortion
    FROM camera c
    JOIN image i ON c.token = i.camera_token
    JOIN lidar_pc lp ON i.timestamp BETWEEN lp.timestamp - 50000 AND lp.timestamp + 50000
    WHERE lp.token IN ({','.join(['?'] * len(lidar_tokens))})
    ORDER BY i.timestamp
    """

    cursor.execute(query, [bytearray.fromhex(t) for t in lidar_tokens])
    rows = cursor.fetchall()

    cameras_data = defaultdict(list)
    for row in rows:
        channel, token, filename, timestamp, ego_pose_token, translation, rotation, intrinsic, distortion = row
        translation = pickle.loads(translation)
        rotation = pickle.loads(rotation)
        intrinsic = pickle.loads(intrinsic)
        distortion = pickle.loads(distortion)

        cameras_data[channel].append({
            'token': token.hex(),
            'file_path': filename,
            'timestamp': timestamp,
            'ego_pose_token': ego_pose_token.hex(),
            'intrinsic': list(intrinsic),
            'distortion': list(distortion),
            'sensor_to_ego_rot': list(rotation),
            'sensor_to_ego_tran': list(translation),
        })

    return cameras_data


def find_nearest_lidar_frame(lidar_data: List[Dict[str, Any]], target_timestamp: int, start_index: int) -> Tuple[Dict[str, Any], int]:
    """
    Locates the nearest LiDAR frame to a given target timestamp in the NuPlan dataset.

    Algorithm:
    1. Start from the given index in the LiDAR data list.
    2. Check a window of 7 frames (3 before, current, 3 after) around the start index.
    3. Calculate the time difference between each frame's timestamp and the target.
    4. Select the LiDAR "frame" with the smallest absolute time difference.

    Args:
        lidar_data: A list of dictionaries, each representing a LiDAR frame.
                    Each dictionary should contain at least a 'timestamp' key.
        target_timestamp: The timestamp to match, typically from a camera frame.
        start_index: The index in lidar_data to start the search from. This is
                     usually the index of the last matched LiDAR frame, to optimize
                     the search for sequential processing.

    Returns:
        A tuple containing two elements:
        1. The dictionary of the nearest LiDAR frame.
        2. The index of this frame in the lidar_data list.

    Note:
        - The function checks 3 LiDAR frames before and after the start index,
          assuming a higher frequency of LiDAR data compared to camera data.
        - This approach is more efficient than searching the entire LiDAR dataset
          for each target timestamp, especially for large datasets.
        - The function assumes that LiDAR data is roughly time-ordered.
    """
    best_frame = None
    best_diff = float('inf')
    best_index = start_index

    for offset in range(-3, 4):  # Check 3 lidar frames before and after (lidar is 20Hz)
        check_index = start_index + offset
        if 0 <= check_index < len(lidar_data):
            frame = lidar_data[check_index]
            diff = abs(frame['timestamp'] - target_timestamp)
            if diff < best_diff:
                best_frame = frame
                best_diff = diff
                best_index = check_index

    return best_frame, best_index

def process_db_file_scenario(db_file: str, sensor_root: str, lidar_tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Processes a single database file from the NuPlan dataset to extract and synchronize sensor data.

    Detailed Process:
    1. Database Connection:
       - Establishes a connection to the SQLite database file.
       - Creates a cursor for executing SQL queries.

    2. Data Extraction:
       - Fetches LiDAR data using the `fetch_lidar_data` function.
       - Retrieves camera data for all channels using `fetch_camera_data`.
       - Extracts ego vehicle pose information with `fetch_ego_poses`.

    3. Frame Synchronization:
       - For each set of synchronized camera frames:
         a. Calculates the average timestamp of the camera frames.
         b. Finds the nearest LiDAR frame to this average timestamp using `find_nearest_lidar_frame`.
         c. Retrieves the corresponding ego pose for the LiDAR frame.

    4. Data Structuring:
       - For each synchronized set of sensor data, creates a comprehensive dictionary containing:
         a. Scene information (e.g., database file name).
         b. Ego state (position, orientation, velocity, acceleration).
         c. LiDAR data (file path, timestamp, sensor-to-ego transformations).
         d. Camera data for each channel (file path, timestamp, calibration parameters).

    Args:
        db_file: Path to the SQLite database file representing a single log in the NuPlan dataset.
        sensor_root: Root directory of the sensor data of the dataset.

    Returns:
        A list of dictionaries, where each dictionary represents a synchronized set of
        sensor data (LiDAR, multiple cameras, ego pose) for a single timestamp. The structure is:
        [
            {
                'scene': {'name': '...'},
                'ego_state': {'ego_center': [...], 'ego_heading': [...], ...},
                'LIDAR_TOP': {'file_path': '...', 'timestamp': ..., ...},
                'CAM_F0': {'file_path': '...', 'timestamp': ..., ...},
                'CAM_L0': {...},
                ...
            },
            ...  # More synchronized frames
        ]

    Raises:
        sqlite3.Error: If there's an issue accessing or querying the database.
        ValueError: If there's an error in data synchronization or processing.
    """
    with get_db_cursor(db_file) as cursor:
        lidar_data = fetch_lidar_data_scenario(cursor, lidar_tokens)
        camera_data = fetch_camera_data_scenario(cursor, lidar_tokens)
        ego_poses = fetch_ego_poses_scenario(cursor, lidar_tokens)

    # Change from {cam_name: [temporal list of cam_info]} to [temporal list of {cam_name, cam_info_i}]
    synchronized_frames = [{cam_name: cam[i] for cam_name, cam in camera_data.items()} for i in range(len(camera_data["CAM_F0"]))]

    synchronized_log_data = []
    frame_with_invalid_files = 0
    last_lidar_index = 0

    for frame_set in synchronized_frames:
        avg_timestamp = int(np.mean([frame['timestamp'] for frame in frame_set.values()]))
        nearest_lidar, last_lidar_index = find_nearest_lidar_frame(lidar_data, avg_timestamp, last_lidar_index)
        ego_pose = ego_poses[nearest_lidar['ego_pose_token']]

        is_valid = True
        if not os.path.exists(os.path.join(sensor_root, nearest_lidar['filename'])):
            is_valid = False
            # print(f"Lidar file not found: {os.path.join(sensor_root, nearest_lidar['filename'])}")


        frame_data = {
            'scene': {
                'name': os.path.basename(db_file),
            },
            'ego_state': {
                'ego_center': ego_pose['position'],
                'ego_heading': ego_pose['orientation'],
                'ego_velocity': ego_pose['velocity'],
                'ego_acceleration': ego_pose['acceleration'],
            },
            'LIDAR_TOP': {
                'file_path': nearest_lidar['filename'],
                'timestamp': nearest_lidar['timestamp'],
                'sensor_to_ego_rot': nearest_lidar['sensor_to_ego_rotation'],
                'sensor_to_ego_tran': nearest_lidar['sensor_to_ego_translation'],
                'ego_to_world_rot': ego_pose['orientation'],
                'ego_to_world_tran': ego_pose['position'],
            },
        }

        for channel, camera_frame in frame_set.items():
            if not os.path.exists(os.path.join(sensor_root, camera_frame['file_path'])):
                is_valid = False
                print(f"Camera file not found: {os.path.join(sensor_root, camera_frame['file_path'])}")
            frame_data[channel] = {
                'file_path': camera_frame['file_path'],
                'timestamp': camera_frame['timestamp'],
                'intrinsic': camera_frame['intrinsic'],
                'distortion': camera_frame['distortion'],
                'sensor_to_ego_rot': camera_frame['sensor_to_ego_rot'],
                'sensor_to_ego_tran': camera_frame['sensor_to_ego_tran'],
                'ego_to_world_rot': ego_pose['orientation'],
                'ego_to_world_tran': ego_pose['position'],
            }
        if not is_valid:
            frame_with_invalid_files += 1
            continue

        synchronized_log_data.append(frame_data)
    #print(f"valid/invalid frames: {len(synchronized_log_data)}/{frame_with_invalid_files}")
    return synchronized_log_data


CAMERAS_LIST = [
    'CAM_L0',
    'CAM_F0',
    'CAM_R0',
    'CAM_R1',
    'CAM_R2',
    'CAM_B0',
    'CAM_L2',
    'CAM_L1'
]

def verify_10hz_synchronization(synchronized_log_data: List[Dict[str, Any]], db_file: str) -> None:
    """
    This function checks the timestamps of each camera's data to ensure
    that they are consistently spaced at approximately 0.1 second intervals.

    Args:
        synchronized_log_data: List of synchronized sensor data.
        db_file: Path to the database file (for error reporting).

    Returns:
        A boolean indicating if the data is properly synchronized at 10Hz.

    Note:
        The function allows for a 0.05 second margin of error in the synchronization.
    """

    if len(synchronized_log_data) == 0:
        raise ValueError(f"Synchronized log data is empty for database file {db_file}")

    for sensor in CAMERAS_LIST:
        sensor_timestamps = [frame[sensor]['timestamp'] for frame in synchronized_log_data]

        if len(sensor_timestamps) <= 1:
            raise ValueError(f"Sensor timestamps empty for sensor {sensor} with database file {db_file}")

        time_differences = np.diff(np.array(sensor_timestamps))
        max_diff = abs(time_differences - 100000).max() # max diff above 10Hz

        if max_diff > 50000:  # Check if sensor data is correctly extracted at 10Hz with a 0.05s margin of error
            error_index = np.argmax(abs(time_differences - 100000))
            error_timestamp = sensor_timestamps[error_index]
            print(
                f"Sensor synchronization error in database file {db_file}:\n"
                f"  Sensor: {sensor}\n"
                f"  Error occurred at timestamp: {error_timestamp}\n"
                f"  Timestamp index: {error_index}\n"
                f"  Measured time difference: {max_diff} microseconds\n"
                f"  Number of camera frames in db files: {len(synchronized_log_data)}"
            )
            return False

    return True


