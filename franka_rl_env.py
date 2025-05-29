"""
Defines the FrankaShelfEnv, a vectorized reinforcement learning environment
for a Franka Emika Panda robot interacting with a shelf.

The environment involves tasks like reaching specific poses within or around a
dynamically configured shelf, considering obstacles and robot joint limits.
It uses the Genesis simulation engine for physics and rendering.
The observation space includes robot state, target position, and a voxelized
representation of obstacles. The action space is continuous joint velocity control.
"""
import os
import random
import math
from typing import List, Any, Optional, Union, Type, Dict, Tuple
import traceback  # For more detailed error messages if needed

import gymnasium as gym  # For type hinting with gym.Wrapper
from gymnasium import spaces
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

import genesis as gs  # type: ignore
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices

# --- Utility Functions ---


def to_numpy(data: Union[torch.Tensor, np.ndarray, List, Tuple]) -> np.ndarray:
    """
    Converts input data (PyTorch Tensor, list, tuple) to a NumPy array.

    If the input is a PyTorch Tensor on a CUDA device, it's moved to the CPU
    before conversion.

    :param data: The data to convert.
    :return: A NumPy array representation of the input data.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy() if data.is_cuda else data.numpy()
    return np.asarray(data, dtype=np.float32)


def transform_point_to_local_frame(
    world_point: np.ndarray,
    frame_world_pos: np.ndarray,
    frame_world_orient_quat_wxyz: np.ndarray  # Expects [w, x, y, z]
) -> np.ndarray:
    """
    Transforms a point from world coordinates to a local frame's coordinates
    using scipy for rotation.

    :param world_point: The point(s) in world coordinates. Shape (3,) or (N, 3).
    :param frame_world_pos: The origin of the local frame in world. Shape (3,).
    :param frame_world_orient_quat_wxyz: WXYZ quaternion of local frame in world. Shape (4,).
    :return: The point(s) in local frame coordinates. Shape (3,) or (N, 3).
    """
    single_op = world_point.ndim == 1
    world_point_b = world_point.reshape(1, -1) if single_op else world_point
    vec_world_b = world_point_b - frame_world_pos
    q_xyzw = frame_world_orient_quat_wxyz[[1, 2, 3, 0]]  # XYZW for SciPy
    rotation = ScipyRotation.from_quat(q_xyzw)
    R_local_from_world = rotation.as_matrix().T
    local_points_arr = (R_local_from_world @ vec_world_b.T).T
    return local_points_arr[0] if single_op else local_points_arr


def _is_point_inside_box(
    point_local: np.ndarray,
    box_half_sizes: np.ndarray
) -> bool:
    """Checks if a point (local coords) is inside a box."""
    return np.all(np.abs(point_local) <= box_half_sizes)


# --- NumPy-based Voxelization Helper Functions ---

def boxes_to_voxel_grid(
    list_of_box_params: List[Dict[str, np.ndarray]],
    grid_dims: Tuple[int, int, int],
    grid_origin_world: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """
    Voxelizes a list of boxes into a 3D grid using NumPy.

    Each box is defined by its world position, size (full dimensions), and
    world orientation (WXYZ quaternion). A voxel is marked as occupied (1.0)
    if its center falls within any of the boxes.

    :param list_of_box_params: A list of dictionaries, each containing:
                               'position': np.ndarray (3,) - box center in world.
                               'size': np.ndarray (3,) - box full dimensions (width, depth, height).
                               'orientation_quat': np.ndarray (4,) - box WXYZ orientation in world.
    :param grid_dims: Tuple (Depth, Height, Width) - dimensions of the voxel grid in number of voxels.
    :param grid_origin_world: np.ndarray (3,) - world coordinates of the grid's corner (min x, y, z).
    :param voxel_size: float - The side length of a single cubic voxel.
    :return: A 3D NumPy array (Depth, Height, Width) representing the voxel grid.
    """
    grid = np.zeros(grid_dims, dtype=np.float32)
    grid_D_elements, grid_H_elements, grid_W_elements = grid_dims
    origin_x, origin_y, origin_z = grid_origin_world

    if not list_of_box_params:
        return grid

    box_half_sizes_list = [box_params['size'] /
                           2.0 for box_params in list_of_box_params]

    for k_depth_idx in range(grid_D_elements):
        for j_height_idx in range(grid_H_elements):
            for i_width_idx in range(grid_W_elements):
                voxel_center_world_x = origin_x + \
                    (i_width_idx + 0.5) * voxel_size
                voxel_center_world_y = origin_y + \
                    (j_height_idx + 0.5) * voxel_size
                voxel_center_world_z = origin_z + \
                    (k_depth_idx + 0.5) * voxel_size
                voxel_center_world = np.array(
                    [voxel_center_world_x, voxel_center_world_y, voxel_center_world_z], dtype=np.float32)

                for box_idx, box_params in enumerate(list_of_box_params):
                    box_world_pos = box_params['position']
                    box_world_orient_quat_wxyz = box_params['orientation_quat']
                    box_half_sizes = box_half_sizes_list[box_idx]

                    voxel_center_local = transform_point_to_local_frame(
                        voxel_center_world, box_world_pos, box_world_orient_quat_wxyz
                    )

                    if _is_point_inside_box(voxel_center_local, box_half_sizes):
                        grid[k_depth_idx, j_height_idx, i_width_idx] = 1.0
                        break
    return grid


def boxes_to_voxel_grid_batched(
    list_of_batched_box_params: List[Dict[str, Any]],
    grid_dims: Tuple[int, int, int],
    grid_origins_batch_world: np.ndarray,
    voxel_size: float,
    num_envs: int
) -> np.ndarray:
    """
    Voxelizes boxes for a batch of environments using NumPy.

    This function iterates through each environment in the batch and calls
    `boxes_to_voxel_grid` for each one.

    :param list_of_batched_box_params: List of dictionaries. Each dict represents a type of box component.
                                       'position': np.ndarray (num_envs, 3) - box centers in world for each env.
                                       'size': np.ndarray (3,) - box full dimensions (same for all envs for this component).
                                       'orientation_quat': np.ndarray (num_envs, 4) - box WXYZ orientations for each env.
    :param grid_dims: Tuple (Depth, Height, Width) - dimensions of the voxel grid.
    :param grid_origins_batch_world: np.ndarray (num_envs, 3) - world coordinates of grid corner for each env.
    :param voxel_size: float - The side length of a single cubic voxel.
    :param num_envs: int - The number of parallel environments in the batch.
    :return: A 4D NumPy array (num_envs, Depth, Height, Width) for the batched voxel grids.
    """
    batched_grids = np.zeros((num_envs,) + grid_dims, dtype=np.float32)
    for env_idx in range(num_envs):
        current_env_box_params = []
        for component_params in list_of_batched_box_params:
            current_env_box_params.append({
                'position': component_params['position'][env_idx],
                'size': component_params['size'],
                'orientation_quat': component_params['orientation_quat'][env_idx]
            })
        batched_grids[env_idx] = boxes_to_voxel_grid(
            current_env_box_params, grid_dims, grid_origins_batch_world[env_idx], voxel_size
        )
    return batched_grids


# --- FrankaShelfEnv Class ---
class FrankaShelfEnv(VecEnv):
    """
    A vectorized environment for a Franka Emika Panda robot interacting with a shelf.
    (Docstring content remains the same as previous version)
    """
    FRANKA_JOINT_NAMES: List[str] = ['joint1', 'joint2', 'joint3', 'joint4',
                                     'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
    FRANKA_NUM_ARM_JOINTS: int = 7
    FRANKA_NUM_TOTAL_JOINTS: int = len(FRANKA_JOINT_NAMES)
    FRANKA_DEFAULT_INITIAL_QPOS: np.ndarray = np.array(
        [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04], dtype=np.float32)
    FRANKA_QPOS_LOWER: np.ndarray = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    FRANKA_QPOS_UPPER: np.ndarray = np.array(
        [2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)
    FRANKA_VEL_LIMIT: np.ndarray = np.array(
        [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float32)
    ROBOT_EE_LINK_NAME: str = "hand"
    DEFAULT_PLATE_WIDTH: float = 0.5
    DEFAULT_PLATE_DEPTH: float = 0.4
    DEFAULT_PLATE_THICKNESS: float = 0.02
    DEFAULT_WALL_THICKNESS: float = 0.02
    DEFAULT_ACTUAL_OPENING_HEIGHT: float = 0.25
    DEFAULT_WALL_COMPONENT_HEIGHT: float = DEFAULT_ACTUAL_OPENING_HEIGHT
    COMPONENT_SIZE_PLATE: np.ndarray = np.array(
        [DEFAULT_PLATE_WIDTH, DEFAULT_PLATE_DEPTH, DEFAULT_PLATE_THICKNESS], dtype=np.float32)
    COMPONENT_SIZE_SIDE_WALL: np.ndarray = np.array(
        [DEFAULT_WALL_THICKNESS, DEFAULT_PLATE_DEPTH, DEFAULT_WALL_COMPONENT_HEIGHT], dtype=np.float32)
    COMPONENT_SIZE_BACK_WALL: np.ndarray = np.array(
        [DEFAULT_PLATE_WIDTH, DEFAULT_WALL_THICKNESS, DEFAULT_WALL_COMPONENT_HEIGHT], dtype=np.float32)
    FIXED_COMPONENT_SIZES: List[np.ndarray] = [
        COMPONENT_SIZE_PLATE, COMPONENT_SIZE_PLATE,
        COMPONENT_SIZE_SIDE_WALL, COMPONENT_SIZE_SIDE_WALL, COMPONENT_SIZE_BACK_WALL
    ]
    SHELF_NUM_COMPONENTS: int = len(FIXED_COMPONENT_SIZES)

    def __init__(self,
                 render_mode: Optional[str] = None,
                 num_envs: int = 1,
                 env_spacing: Tuple[float, float] = (2.0, 2.0),
                 voxel_grid_dims: Tuple[int, int, int] = (32, 48, 48),
                 workspace_bounds_xyz: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
                     (-0.8, 0.8), (-0.8, 0.8), (0.0, 1.5)),
                 voxel_grid_world_size: Tuple[float,
                                              float, float] = (0.8, 0.8, 0.5),
                 max_steps_per_episode: int = 1000,
                 dt: float = 0.01,
                 franka_xml_path: str = 'xml/franka_emika_panda/panda.xml',
                 k_dist_reward: float = 1.0,
                 k_time_penalty: float = 0.01,
                 k_action_penalty: float = 0.001,
                 k_joint_limit_penalty: float = 10.0,
                 k_collision_penalty: float = 100.0,
                 k_accel_penalty: float = 0.0,
                 success_reward_val: float = 200.0,
                 success_threshold_val: float = 0.05,
                 video_camera_pos: Tuple[float,
                                         float, float] = (1.8, -1.8, 2.0),
                 video_camera_lookat: Tuple[float,
                                            float, float] = (0.3, 0.0, 0.5),
                 video_camera_fov: float = 45,
                 video_res: Tuple[int, int] = (320, 240)
                 ):
        self.render_mode = render_mode
        self._num_envs = num_envs
        robot_state_flat_dim = self.FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4
        relative_target_pos_dim = 3
        _observation_space = spaces.Dict({
            "robot_state_flat": spaces.Box(
                low=np.full(robot_state_flat_dim, -np.inf, dtype=np.float32),
                high=np.full(robot_state_flat_dim, np.inf, dtype=np.float32),
                dtype=np.float32),
            "relative_target_pos": spaces.Box(
                low=np.full(relative_target_pos_dim, -
                            np.inf, dtype=np.float32),
                high=np.full(relative_target_pos_dim,
                             np.inf, dtype=np.float32),
                dtype=np.float32),
            "obstacle_voxels": spaces.Box(
                low=0, high=1, shape=voxel_grid_dims, dtype=np.float32)
        })
        _action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.FRANKA_NUM_ARM_JOINTS,), dtype=np.float32)
        super().__init__(num_envs=self._num_envs,
                         observation_space=_observation_space, action_space=_action_space)
        self.env_spacing = env_spacing
        self.voxel_grid_dims = voxel_grid_dims
        self.workspace_bounds_x, self.workspace_bounds_y, self.workspace_bounds_z = workspace_bounds_xyz
        self.voxel_grid_world_span_x, self.voxel_grid_world_span_y, self.voxel_grid_world_span_z = voxel_grid_world_size
        _vx_span_per_voxel = self.voxel_grid_world_span_x / \
            self.voxel_grid_dims[2]
        _vy_span_per_voxel = self.voxel_grid_world_span_y / \
            self.voxel_grid_dims[1]
        _vz_span_per_voxel = self.voxel_grid_world_span_z / \
            self.voxel_grid_dims[0]
        self.voxel_size = min(_vx_span_per_voxel,
                              _vy_span_per_voxel, _vz_span_per_voxel)
        eff_grid_span_x = self.voxel_grid_dims[2] * self.voxel_size
        eff_grid_span_y = self.voxel_grid_dims[1] * self.voxel_size
        self.grid_origin_world_x = self.workspace_bounds_x[0] + \
            ((self.workspace_bounds_x[1] -
             self.workspace_bounds_x[0]) - eff_grid_span_x) / 2.0
        self.grid_origin_world_y = self.workspace_bounds_y[0] + \
            ((self.workspace_bounds_y[1] -
             self.workspace_bounds_y[0]) - eff_grid_span_y) / 2.0
        self.grid_origin_world_z = self.workspace_bounds_z[0]
        self.grid_origin_global_frame = np.array(
            [self.grid_origin_world_x, self.grid_origin_world_y, self.grid_origin_world_z], dtype=np.float32)
        self.grid_origins_batch_world = np.tile(
            self.grid_origin_global_frame, (self.num_envs, 1))
        self.max_steps = max_steps_per_episode
        self.current_step_per_env = np.zeros(self.num_envs, dtype=np.int32)
        self.dt = dt
        self.franka_xml_path = franka_xml_path
        self.k_dist_reward = k_dist_reward
        self.k_time_penalty = k_time_penalty
        self.k_action_penalty = k_action_penalty
        self.k_joint_limit_penalty = k_joint_limit_penalty
        self.k_collision_penalty = k_collision_penalty
        self.k_accel_penalty = k_accel_penalty
        self.success_reward = success_reward_val
        self.success_threshold = success_threshold_val
        self._define_shelf_configurations()
        self.current_shelf_config_key_per_env: List[str] = [""] * self.num_envs
        self.current_shelf_instance_params_per_env: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)]
        self.metadata = {'render_modes': [
            'human', 'rgb_array', 'video'], 'render_fps': 60}
        sim_viewer_options = gs.options.ViewerOptions(
            camera_pos=video_camera_pos,
            camera_lookat=video_camera_lookat,
            camera_fov=video_camera_fov,
            res=video_res if self.render_mode == "human" else (
                960, 640),  # type: ignore
            max_FPS=self.metadata['render_fps']
        )
        self.scene = gs.Scene(
            viewer_options=sim_viewer_options,
            sim_options=gs.options.SimOptions(dt=self.dt),
            show_viewer=(self.render_mode == "human")  # type: ignore
        )
        self.video_camera_params = {
            'pos': video_camera_pos, 'lookat': video_camera_lookat,
            'fov': video_camera_fov, 'res': video_res
        }
        self._is_recording_active = False
        self.video_capture_camera = self.scene.add_camera(
            res=self.video_camera_params['res'],
            pos=self.video_camera_params['pos'],
            lookat=self.video_camera_params['lookat'],
            fov=self.video_camera_params['fov'],
            GUI=False
        )
        self.plane_entity = self.scene.add_entity(gs.morphs.Plane())
        self.franka_entity = self.scene.add_entity(
            gs.morphs.MJCF(file=self.franka_xml_path))
        self.shelf_component_entities: List[Any] = [
            self.scene.add_entity(gs.morphs.Box(
                pos=(0, -10 - i * 0.5, 0),
                quat=(1, 0, 0, 0),
                size=tuple(s),
                fixed=True,
                visualization=True,
                collision=True))
            for i, s in enumerate(self.FIXED_COMPONENT_SIZES)
        ]
        if not self.scene.is_built:
            self.scene.build(n_envs=self.num_envs,
                             env_spacing=self.env_spacing)

        self.target_position_world_batch = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        self.current_voxel_grid_batch = np.zeros(
            (self.num_envs,) + self.voxel_grid_dims, dtype=np.float32)
        self.prev_joint_vel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)
        self.franka_all_dof_indices_local = np.array([self.franka_entity.get_joint(name).dof_idx_local
                                                      for name in self.FRANKA_JOINT_NAMES], dtype=np.int32)
        self.franka_arm_dof_indices_local = self.franka_all_dof_indices_local[
            :self.FRANKA_NUM_ARM_JOINTS]
        self.max_allowable_joint_velocity_scale = 0.5
        self.actions_buffer = np.zeros(
            (self.num_envs, self.action_space.shape[0]), dtype=self.action_space.dtype)  # type: ignore
        self.buf_infos: List[Dict[str, Any]] = [{}
                                                for _ in range(self.num_envs)]

    def _define_shelf_configurations(self) -> None:
        """Defines various preset configurations for the shelf's position and orientation."""
        self.shelf_configurations: Dict[str, Dict[str, Any]] = {
            "high_center_reach": {
                "name": "high_center_reach",
                "base_pos_range_x": (0.45, 0.65), "base_pos_range_y": (-0.2, 0.2),
                "base_pos_range_z": (0.5, 0.65)
            },
            "low_forward_reach": {
                "name": "low_forward_reach",
                "base_pos_range_x": (0.40, 0.60), "base_pos_range_y": (-0.2, 0.2),
                "base_pos_range_z": (0.05, 0.20)
            },
            "mid_side_reach_right": {
                "name": "mid_side_reach_right",
                "base_pos_range_x": (0.25, 0.5), "base_pos_range_y": (0.4, 0.6),
                "base_pos_range_z": (0.25, 0.45)
            },
            "mid_side_reach_left": {
                "name": "mid_side_reach_left",
                "base_pos_range_x": (0.25, 0.5), "base_pos_range_y": (-0.6, -0.4),
                "base_pos_range_z": (0.25, 0.45)
            }
        }
        self.shelf_config_keys_list: List[str] = list(
            self.shelf_configurations.keys())

    def _get_robot_state_parts_batched(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retrieves batched robot state information from the Genesis simulation."""
        joint_pos_all_batch = to_numpy(
            self.franka_entity.get_dofs_position(self.franka_all_dof_indices_local))
        joint_vel_all_batch = to_numpy(
            self.franka_entity.get_dofs_velocity(self.franka_all_dof_indices_local))
        arm_joint_pos_batch = joint_pos_all_batch[:,
                                                  :self.FRANKA_NUM_ARM_JOINTS]
        arm_joint_vel_batch = joint_vel_all_batch[:,
                                                  :self.FRANKA_NUM_ARM_JOINTS]
        ee_link = self.franka_entity.get_link(self.ROBOT_EE_LINK_NAME)
        ee_pos_batch = to_numpy(ee_link.get_pos())
        ee_orient_quat_wxyz_batch = to_numpy(ee_link.get_quat())
        return arm_joint_pos_batch, arm_joint_vel_batch, ee_pos_batch, ee_orient_quat_wxyz_batch

    def _build_shelf_structure_batched(self) -> List[Dict[str, Any]]:
        """Constructs the shelf in each parallel environment with randomized configurations."""
        shelf_assembly_origins_world_batch = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        shelf_assembly_yaws_batch = np.zeros(self.num_envs, dtype=np.float32)

        for i in range(self.num_envs):
            config_key = random.choice(self.shelf_config_keys_list)
            self.current_shelf_config_key_per_env[i] = config_key
            config = self.shelf_configurations[config_key]
            shelf_assembly_origins_world_batch[i, 0] = random.uniform(
                *config["base_pos_range_x"])
            shelf_assembly_origins_world_batch[i, 1] = random.uniform(
                *config["base_pos_range_y"])
            shelf_assembly_origins_world_batch[i, 2] = random.uniform(
                *config["base_pos_range_z"])
            if "center" in config["name"] or "forward" in config["name"]:
                shelf_assembly_yaws_batch[i] = math.atan2(
                    -shelf_assembly_origins_world_batch[i, 1], -shelf_assembly_origins_world_batch[i, 0]) - np.pi / 2.0
            elif "right" in config["name"]:
                shelf_assembly_yaws_batch[i] = np.pi
            elif "left" in config["name"]:
                shelf_assembly_yaws_batch[i] = 0.0
            else:
                shelf_assembly_yaws_batch[i] = 0.0
            self.current_shelf_instance_params_per_env[i] = {
                "shelf_assembly_origin_world": shelf_assembly_origins_world_batch[i].copy(),
                "shelf_assembly_yaw": shelf_assembly_yaws_batch[i],
                "actual_opening_height": self.DEFAULT_ACTUAL_OPENING_HEIGHT,
                "internal_width": self.DEFAULT_PLATE_WIDTH - 2 * self.DEFAULT_WALL_THICKNESS,
                "internal_depth": self.DEFAULT_PLATE_DEPTH - self.DEFAULT_WALL_THICKNESS,
            }

        cos_yaw_half_batch = np.cos(shelf_assembly_yaws_batch / 2.0)
        sin_yaw_half_batch = np.sin(shelf_assembly_yaws_batch / 2.0)
        shelf_assembly_world_quats_batch_wxyz = np.zeros(
            (self.num_envs, 4), dtype=np.float32)
        shelf_assembly_world_quats_batch_wxyz[:, 0] = cos_yaw_half_batch
        shelf_assembly_world_quats_batch_wxyz[:, 3] = sin_yaw_half_batch

        shelf_assembly_world_quats_batch_xyzw = np.zeros(
            (self.num_envs, 4), dtype=np.float32)
        shelf_assembly_world_quats_batch_xyzw[:,
                                              0] = shelf_assembly_world_quats_batch_wxyz[:, 1]
        shelf_assembly_world_quats_batch_xyzw[:,
                                              1] = shelf_assembly_world_quats_batch_wxyz[:, 2]
        shelf_assembly_world_quats_batch_xyzw[:,
                                              2] = shelf_assembly_world_quats_batch_wxyz[:, 3]
        shelf_assembly_world_quats_batch_xyzw[:,
                                              3] = shelf_assembly_world_quats_batch_wxyz[:, 0]

        bottom_plate_s, top_plate_s, side_wall_s, _, back_wall_s = self.FIXED_COMPONENT_SIZES[
            0:5]
        actual_opening_h = self.DEFAULT_ACTUAL_OPENING_HEIGHT
        local_component_offsets = [
            np.array([0, 0, bottom_plate_s[2] / 2.0]),
            np.array([0, 0, bottom_plate_s[2] +
                     actual_opening_h + top_plate_s[2] / 2.0]),
            np.array([-(bottom_plate_s[0] / 2.0 - side_wall_s[0] / 2.0), 0,
                      bottom_plate_s[2] + side_wall_s[2] / 2.0]),
            np.array([(bottom_plate_s[0] / 2.0 - side_wall_s[0] / 2.0), 0,
                      bottom_plate_s[2] + side_wall_s[2] / 2.0]),
            np.array([0, -(bottom_plate_s[1] / 2.0 - back_wall_s[1] / 2.0),
                      bottom_plate_s[2] + back_wall_s[2] / 2.0])
        ]
        list_of_box_params_for_voxelization_batched: List[Dict[str, Any]] = []

        for comp_idx, component_entity in enumerate(self.shelf_component_entities):
            local_offset_of_this_component = local_component_offsets[comp_idx]
            fixed_size_of_this_component = self.FIXED_COMPONENT_SIZES[comp_idx]
            batched_final_comp_pos_world = np.zeros(
                (self.num_envs, 3), dtype=np.float32)
            for env_i in range(self.num_envs):
                R_shelf_assembly_yaw_i = ScipyRotation.from_quat(
                    shelf_assembly_world_quats_batch_xyzw[env_i]).as_matrix()
                rotated_offset_world_i = R_shelf_assembly_yaw_i @ local_offset_of_this_component
                batched_final_comp_pos_world[env_i] = shelf_assembly_origins_world_batch[env_i] + \
                    rotated_offset_world_i

            pos_tensor = torch.tensor(
                batched_final_comp_pos_world, device=gs.device, dtype=torch.float32)  # type: ignore
            quat_tensor = torch.tensor(
                shelf_assembly_world_quats_batch_wxyz, device=gs.device, dtype=torch.float32)  # type: ignore
            component_entity.set_pos(pos_tensor)
            component_entity.set_quat(quat_tensor)
            list_of_box_params_for_voxelization_batched.append({
                'position': batched_final_comp_pos_world,
                'size': fixed_size_of_this_component,
                'orientation_quat': shelf_assembly_world_quats_batch_wxyz
            })
        return list_of_box_params_for_voxelization_batched

    def _generate_target_in_shelf_batched(self) -> np.ndarray:
        """Generates a target position within the usable volume of the shelf for each environment."""
        self.target_position_world_batch.fill(0.0)
        for i in range(self.num_envs):
            params = self.current_shelf_instance_params_per_env[i]
            open_w, open_d, open_h = params["internal_width"], params["internal_depth"], params["actual_opening_height"]
            target_local_x = random.uniform(-open_w /
                                            2 * 0.7, open_w / 2 * 0.7)
            target_local_y = random.uniform(-open_d /
                                            2 * 0.7, open_d / 2 * 0.3)
            target_local_z = random.uniform(-open_h /
                                            2 * 0.8, open_h / 2 * 0.8)
            target_in_shelf_opening_frame = np.array(
                [target_local_x, target_local_y, target_local_z], dtype=np.float32)

            shelf_assembly_origin_world = params["shelf_assembly_origin_world"]
            shelf_assembly_yaw = params["shelf_assembly_yaw"]
            offset_to_opening_center_local = np.array(
                [0, 0, self.DEFAULT_PLATE_THICKNESS + open_h / 2.0], dtype=np.float32)

            shelf_assembly_quat_world_wxyz = np.array(
                [np.cos(shelf_assembly_yaw / 2.0), 0, 0, np.sin(shelf_assembly_yaw / 2.0)], dtype=np.float32)
            shelf_assembly_quat_world_xyzw = shelf_assembly_quat_world_wxyz[[
                1, 2, 3, 0]]  # XYZW for SciPy

            R_shelf_yaw_world = ScipyRotation.from_quat(
                shelf_assembly_quat_world_xyzw).as_matrix()
            opening_center_world = shelf_assembly_origin_world + \
                (R_shelf_yaw_world @ offset_to_opening_center_local)
            target_offset_world_rotated = R_shelf_yaw_world @ target_in_shelf_opening_frame
            self.target_position_world_batch[i] = opening_center_world + \
                target_offset_world_rotated

        if self.render_mode == "human" or self._is_recording_active:  # type: ignore
            if self.scene and hasattr(self.scene, 'clear_debug_objects'):
                self.scene.clear_debug_objects()
            if self.num_envs == 1 and self.scene:
                self.scene.draw_debug_spheres(
                    self.target_position_world_batch.tolist(), radius=0.03, color=(0, 1, 0, 0.8))
        return self.target_position_world_batch

    def _get_obs_batched(self) -> VecEnvObs:
        """Constructs the observation dictionary for all environments."""
        arm_qpos, arm_qvel, ee_pos, ee_quat = self._get_robot_state_parts_batched()
        robot_state_flat_batch = np.concatenate(
            [arm_qpos, arm_qvel, ee_pos, ee_quat], axis=1).astype(np.float32)
        relative_target_pos_batch = (
            self.target_position_world_batch - ee_pos).astype(np.float32)
        return {
            "robot_state_flat": robot_state_flat_batch,
            "relative_target_pos": relative_target_pos_batch,
            "obstacle_voxels": self.current_voxel_grid_batch.copy()
        }

    # type: ignore
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> VecEnvObs:
        """Resets all parallel environments to an initial state."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if self.scene:
            if self.scene.is_built:
                self.scene.reset()
            else:
                self.scene.build(n_envs=self.num_envs,
                                 env_spacing=self.env_spacing)
        else:
            raise RuntimeError(
                "Genesis scene is not initialized. Cannot reset.")
        self.current_step_per_env.fill(0)
        initial_qpos_batch = np.tile(
            self.FRANKA_DEFAULT_INITIAL_QPOS, (self.num_envs, 1))
        initial_qvel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_TOTAL_JOINTS), dtype=np.float32)
        qpos_tensor = torch.tensor(
            initial_qpos_batch, device=gs.device, dtype=torch.float32)  # type: ignore
        qvel_tensor = torch.tensor(
            initial_qvel_batch, device=gs.device, dtype=torch.float32)  # type: ignore
        if self.franka_entity:
            self.franka_entity.set_dofs_position(
                qpos_tensor, self.franka_all_dof_indices_local)
            self.franka_entity.set_dofs_velocity(
                qvel_tensor, self.franka_all_dof_indices_local)
        self.prev_joint_vel_batch.fill(0.0)

        # Voxelization step using NumPy-based approach
        list_of_box_params = self._build_shelf_structure_batched()
        self.current_voxel_grid_batch = boxes_to_voxel_grid_batched(
            list_of_box_params,
            self.voxel_grid_dims,
            self.grid_origins_batch_world,
            self.voxel_size,
            self.num_envs
        )

        self._generate_target_in_shelf_batched()
        obs_batch = self._get_obs_batched()
        self.buf_infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.buf_infos[i]["shelf_config"] = self.current_shelf_config_key_per_env[i]
        return obs_batch

    def step_async(self, actions: np.ndarray) -> None:
        """Stores the actions to be executed in the next `step_wait()` call."""
        self.actions_buffer = actions.copy()

    def _calculate_rewards_and_dones(
        self,
        actions_clipped_batch: np.ndarray,
        arm_joint_pos_batch: np.ndarray,
        arm_joint_vel_batch: np.ndarray,
        ee_pos_batch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Calculates rewards, dones, and info dictionaries for the current step."""
        dist_to_target_batch = np.linalg.norm(
            ee_pos_batch - self.target_position_world_batch, axis=1)
        reward_distance_batch = -self.k_dist_reward * dist_to_target_batch
        time_penalty_batch = -self.k_time_penalty * \
            np.ones(self.num_envs, dtype=np.float32)
        action_penalty_batch = -self.k_action_penalty * \
            np.sum(np.square(actions_clipped_batch), axis=1)
        joint_pos_penalty_batch = np.zeros(self.num_envs, dtype=np.float32)
        near_limit_threshold = 0.05
        for i in range(self.FRANKA_NUM_ARM_JOINTS):
            lower_limit_violation = (
                self.FRANKA_QPOS_LOWER[i] + near_limit_threshold) - arm_joint_pos_batch[:, i]
            joint_pos_penalty_batch -= self.k_joint_limit_penalty * \
                np.maximum(0, lower_limit_violation)
            upper_limit_violation = arm_joint_pos_batch[:, i] - (
                self.FRANKA_QPOS_UPPER[i] - near_limit_threshold)
            joint_pos_penalty_batch -= self.k_joint_limit_penalty * \
                np.maximum(0, upper_limit_violation)
        collision_penalty_val_batch = np.zeros(self.num_envs, dtype=np.float32)
        terminated_collision_batch = np.zeros(self.num_envs, dtype=bool)
        if self.scene and self.scene.is_built and self.franka_entity:
            contacts_data = self.franka_entity.get_contacts()
            if contacts_data and 'valid_mask' in contacts_data:
                valid_mask_np = to_numpy(contacts_data['valid_mask'])
                for i_env in range(self.num_envs):
                    if np.any(valid_mask_np[i_env, :]):
                        collision_penalty_val_batch[i_env] -= self.k_collision_penalty
                        terminated_collision_batch[i_env] = True
        success_reward_val_batch = np.zeros(self.num_envs, dtype=np.float32)
        terminated_success_batch = dist_to_target_batch < self.success_threshold
        success_reward_val_batch[terminated_success_batch] = self.success_reward
        accel_penalty_batch = np.zeros(self.num_envs, dtype=np.float32)
        if self.k_accel_penalty > 0 and self.dt > 0:
            joint_accel_batch = (arm_joint_vel_batch -
                                 self.prev_joint_vel_batch) / self.dt
            accel_penalty_batch = -self.k_accel_penalty * \
                np.sum(np.square(joint_accel_batch), axis=1)
        rewards_batch = (reward_distance_batch + time_penalty_batch + action_penalty_batch +
                         joint_pos_penalty_batch + collision_penalty_val_batch +
                         success_reward_val_batch + accel_penalty_batch)
        terminated_batch = np.logical_or(
            terminated_collision_batch, terminated_success_batch)
        truncated_batch = self.current_step_per_env >= self.max_steps
        dones_batch = np.logical_or(terminated_batch, truncated_batch)
        current_infos: List[Dict[str, Any]] = []
        for i in range(self.num_envs):
            info_dict: Dict[str, Any] = {
                "is_success": terminated_success_batch[i],
                "distance_to_target": float(dist_to_target_batch[i]),
                "collision_detected": terminated_collision_batch[i],
                "current_step": self.current_step_per_env[i],
                "shelf_config": self.current_shelf_config_key_per_env[i]
            }
            if dones_batch[i] and truncated_batch[i] and not terminated_batch[i]:
                info_dict["TimeLimit.truncated"] = True
            current_infos.append(info_dict)
        return rewards_batch, dones_batch, terminated_batch, current_infos

    def step_wait(self) -> VecEnvStepReturn:
        """Executes stored actions and returns results."""
        if not self.scene or not self.franka_entity:
            raise RuntimeError(
                "Environment not properly initialized. Call reset() first.")
        actions_batch = self.actions_buffer
        self.current_step_per_env += 1
        actions_clipped_batch = np.clip(
            actions_batch, -1.0, 1.0).astype(np.float32)
        scaled_target_velocities_batch = actions_clipped_batch * \
            self.max_allowable_joint_velocity_scale
        actions_tensor = torch.tensor(
            scaled_target_velocities_batch, device=gs.device, dtype=torch.float32)  # type: ignore
        self.franka_entity.control_dofs_velocity(
            actions_tensor, self.franka_arm_dof_indices_local)
        self.scene.step()
        if self._is_recording_active and self.video_capture_camera:
            self.video_capture_camera.render()
        arm_qpos_b, arm_qvel_b, ee_pos_b, _ = self._get_robot_state_parts_batched()
        rewards_b, dones_b, terminated_b, self.buf_infos = self._calculate_rewards_and_dones(
            actions_clipped_batch, arm_qpos_b, arm_qvel_b, ee_pos_b
        )
        observations_b = self._get_obs_batched()
        self.prev_joint_vel_batch = arm_qvel_b.copy()
        for i in range(self.num_envs):
            if dones_b[i]:
                terminal_obs_single_env = {k: v[i]
                                           for k, v in observations_b.items()}
                self.buf_infos[i]["terminal_observation"] = terminal_obs_single_env
        return observations_b, rewards_b, dones_b, self.buf_infos

    # type: ignore
    def render(self, mode: Optional[str] = 'human') -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """Renders the environment."""
        if mode == "human" and self.render_mode == "human":
            return None
        elif mode == "rgb_array":
            if not self.video_capture_camera:
                grid_to_render = self.current_voxel_grid_batch[0]
                img_slice = np.sum(grid_to_render, axis=0)
                if np.max(img_slice) > 0:
                    img_slice_norm = np.clip(
                        img_slice / np.max(img_slice), 0, 1)
                else:
                    img_slice_norm = np.zeros_like(img_slice)
                img_slice_uint8 = (img_slice_norm * 255).astype(np.uint8)
                rgb_image = np.stack([img_slice_uint8] * 3, axis=-1)
                return [rgb_image] if self.num_envs > 1 else rgb_image
            try:
                rgb_output, _, _, _ = self.video_capture_camera.render(
                    rgb=True, depth=False, segmentation=False, normal=False)
                if isinstance(rgb_output, list):
                    return rgb_output[0] if rgb_output else None
                return rgb_output
            except Exception as e:
                print(f"Error during camera render: {e}")
                return None
        return None

    def close(self) -> None:
        """Closes the environment and releases resources."""
        if self.video_capture_camera:
            try:
                if hasattr(self.video_capture_camera, 'close') and callable(self.video_capture_camera.close):
                    self.video_capture_camera.close()
            except Exception as e:
                print(f"Error closing video capture camera: {e}")
            finally:
                self.video_capture_camera = None  # type: ignore
        if self.scene:
            try:
                if hasattr(self.scene, 'close') and callable(self.scene.close):
                    self.scene.close()
                elif hasattr(self.scene, 'shutdown') and callable(self.scene.shutdown):
                    self.scene.shutdown()
            except Exception as e:
                print(f"Error closing/shutting down Genesis scene: {e}")
            finally:
                self.scene = None  # type: ignore

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Gets an attribute from the vectorized environment."""
        if hasattr(self, attr_name):
            data = getattr(self, attr_name)
            return [data for _ in self._get_indices(indices)]
        raise AttributeError(
            f"Attribute '{attr_name}' not found in {self.__class__.__name__}")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Sets an attribute in the vectorized environment."""
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Calls a method in the vectorized environment for specified sub-environments."""
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]
        raise AttributeError(
            f"Method '{method_name}' not found in {self.__class__.__name__}")

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Checks if the environment is wrapped with a given wrapper."""
        return [isinstance(self, wrapper_class) for _ in self._get_indices(indices)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:  # type: ignore
        """Seeds the random number generators in all sub-environments."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed for _ in range(self.num_envs)]  # type: ignore

    def start_video_recording(self, env_idx_to_focus: int = 0) -> bool:
        """Starts recording video frames."""
        if not self.video_capture_camera:
            print("Cannot start recording: Video camera not available.")
            self._is_recording_active = False
            return False
        try:
            self.video_capture_camera.start_recording()
            self._is_recording_active = True
            if self.scene and hasattr(self.scene, 'clear_debug_objects'):
                self.scene.clear_debug_objects()
            if self.scene and (self.num_envs == 1 or env_idx_to_focus == 0):
                self.scene.draw_debug_spheres(
                    [self.target_position_world_batch[env_idx_to_focus].tolist()],
                    radius=0.03, color=(0, 1, 0, 0.8)
                )
            return True
        except Exception as e:
            print(f"Error starting video recording: {e}")
            self._is_recording_active = False
            return False

    def stop_video_recording(self, save_dir: str, filename: str, fps: int = 30) -> Optional[str]:
        """Stops video recording and saves the video to a file."""
        if not self._is_recording_active:
            return None
        if not self.video_capture_camera:
            print(
                "Cannot stop recording: Video camera not available but was marked active.")
            self._is_recording_active = False
            return None
        try:
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(save_dir, filename)
            self.video_capture_camera.stop_recording(
                save_to_filename=full_path, fps=fps)
            return full_path
        except Exception as e:
            print(f"Error stopping/saving video recording: {e}")
            return None
        finally:
            self._is_recording_active = False


# --- Main block for testing the environment ---
if __name__ == '__main__':
    print("--- FrankaShelfEnv Test Script ---")
    gs_initialized_in_main = False
    try:
        if not (hasattr(gs, '_is_initialized') and gs._is_initialized):  # type: ignore
            gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
            gs_initialized_in_main = True
        print("Genesis initialized successfully for testing.")
    except Exception as e_init:
        print(
            f"CRITICAL ERROR: Failed to initialize Genesis for testing: {e_init}\nExiting.")
        sys.exit(1)

    NUM_TEST_ENVS = 4
    env = None
    try:
        print(f"Creating FrankaShelfEnv with num_envs={NUM_TEST_ENVS}...")
        env = FrankaShelfEnv(
            num_envs=NUM_TEST_ENVS, render_mode="human",
            k_collision_penalty=150.0, k_accel_penalty=0.001, video_res=(960, 640),
            # to make it easier to detect 2cm shelves
            voxel_grid_dims=(32, 48, 48),
            voxel_grid_world_size=(0.8, 0.8, 0.5)
        )
        print(
            f"FrankaShelfEnv created. Obs Space: {env.observation_space}, Act Space: {env.action_space}")
        print("\nTesting VecEnv reset...")
        obs_batch = env.reset()
        assert isinstance(obs_batch, dict), "Reset should return a dict."
        assert "obstacle_voxels" in obs_batch, "Obs missing 'obstacle_voxels'."
        assert obs_batch['obstacle_voxels'].shape == (NUM_TEST_ENVS,) + env.voxel_grid_dims, \
            f"Voxel grid shape mismatch. Expected {(NUM_TEST_ENVS,) + env.voxel_grid_dims}, got {obs_batch['obstacle_voxels'].shape}"
        print("Reset successful. Voxel grid shape:",
              obs_batch['obstacle_voxels'].shape)
        # Check if any voxels are set
        print("Sample voxel grid (sum):", np.sum(obs_batch['obstacle_voxels']))

        print("\nTesting video recording (briefly)...")
        if env.start_video_recording(env_idx_to_focus=0):
            print("Recording started for 5 steps...")
            for _ in range(5):
                actions_sample = np.array(
                    [env.action_space.sample() for _ in range(env.num_envs)])
                env.step_async(actions_sample)
                _, _, dones_sample, _ = env.step_wait()
                if np.any(dones_sample):
                    env.reset()
            video_path = env.stop_video_recording(
                save_dir="./videos_env_test", filename="test_franka_env_numpy_vox.mp4", fps=15)
            if video_path:
                print(f"Test video saved to {video_path}")
            else:
                print("Test video saving failed/not active.")
        else:
            print("Could not start video recording for test.")

        print(f"\nRunning 1 example episode (batch of {env.num_envs} envs)...")
        current_obs_batch = env.reset()
        episode_rewards_sum = np.zeros(env.num_envs)
        for step_num in range(500):
            actions = np.array([env.action_space.sample()
                               for _ in range(env.num_envs)])
            env.step_async(actions)
            next_obs_batch, rewards_b, dones_b, infos_b = env.step_wait()
            episode_rewards_sum += rewards_b
            if (step_num + 1) % 5 == 0:
                print(
                    f"  Test Step {step_num+1}: Env 0 Rew={rewards_b[0]:.2f}, Done={dones_b[0]}")
                print("  Voxel grid sum:", np.sum(
                    next_obs_batch['obstacle_voxels']))
            if np.all(dones_b):
                break
        print(f"  Test Episode finished. Total rewards: {episode_rewards_sum}")

    except Exception as e_runtime:
        print(f"ERROR during FrankaShelfEnv example usage: {e_runtime}")
        traceback.print_exc()
    finally:
        if env:
            env.close()
        # type: ignore
        if gs_initialized_in_main and hasattr(gs, 'shutdown') and callable(gs.shutdown):
            print("Shutting down Genesis post-testing.")
            gs.shutdown()  # type: ignore
        print("--- FrankaShelfEnv Test Script Finished ---")
