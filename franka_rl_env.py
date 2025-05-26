# <filename>franka_rl_env.py</filename>
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
import math
from typing import List, Any, Optional, Union, Type, Dict, Tuple
import os

import genesis as gs

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices


def to_numpy(data):
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data.cpu().numpy()
        return data.numpy()
    return np.asarray(data)


def quat_to_rotation_matrix(q_wxyz):
    if q_wxyz.ndim == 1:
        w, x, y, z = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
        x2, y2, z2 = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = np.array([
            [1 - 2*(y2 + z2),     2*(xy - wz),     2*(xz + wy)],
            [2*(xy + wz), 1 - 2*(x2 + z2),     2*(yz - wx)],
            [2*(xz - wy),     2*(yz + wx), 1 - 2*(x2 + y2)]
        ], dtype=np.float32)
    elif q_wxyz.ndim == 2:
        raise NotImplementedError(
            "Batched quaternion to rotation matrix not implemented.")
    else:
        raise ValueError(f"Unexpected quaternion shape: {q_wxyz.shape}")
    return R


def transform_point_to_local_frame(world_point, box_world_pos, box_world_orient_quat_wxyz):
    single_op = world_point.ndim == 1
    if single_op:
        world_point_b = world_point.reshape(1, -1)
        box_world_pos_b = box_world_pos.reshape(1, -1)
        box_world_orient_quat_wxyz_b = box_world_orient_quat_wxyz.reshape(
            1, -1)
    else:
        world_point_b = world_point
        num_points = world_point.shape[0]
        box_world_pos_b = np.tile(box_world_pos, (num_points, 1))
        box_world_orient_quat_wxyz_b = np.tile(
            box_world_orient_quat_wxyz, (num_points, 1))

    vec_world_b = world_point_b - box_world_pos_b
    local_points_list = []
    for i in range(world_point_b.shape[0]):
        R_world_from_local_i = quat_to_rotation_matrix(
            box_world_orient_quat_wxyz_b[i])
        R_local_from_world_i = R_world_from_local_i.T
        local_points_list.append(R_local_from_world_i @ vec_world_b[i])
    local_points_arr = np.array(local_points_list, dtype=np.float32)
    return local_points_arr[0] if single_op else local_points_arr


def boxes_to_voxel_grid_batched(list_of_batched_box_params, grid_dims, grid_origins_batch, voxel_size, num_envs):
    batched_grids = np.zeros((num_envs,) + grid_dims, dtype=np.float32)
    for env_idx in range(num_envs):
        current_env_box_params = []
        for comp_params in list_of_batched_box_params:
            current_env_box_params.append({
                'position': comp_params['position'][env_idx],
                'size': comp_params['size'],
                'orientation_quat': comp_params['orientation_quat'][env_idx]
            })
        batched_grids[env_idx] = boxes_to_voxel_grid(
            current_env_box_params, grid_dims, grid_origins_batch[env_idx], voxel_size
        )
    return batched_grids


def boxes_to_voxel_grid(list_of_box_params, grid_dims, grid_origin, voxel_size):
    grid = np.zeros(grid_dims, dtype=np.float32)
    grid_D_elements, grid_H_elements, grid_W_elements = grid_dims
    origin_x, origin_y, origin_z = grid_origin
    if not list_of_box_params:
        return grid
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
                for box_params in list_of_box_params:
                    box_world_pos = box_params['position']
                    box_local_size = box_params['size']
                    box_world_orient_quat_wxyz = box_params['orientation_quat']
                    voxel_center_local = transform_point_to_local_frame(
                        voxel_center_world, box_world_pos, box_world_orient_quat_wxyz
                    )
                    half_size_local_x, half_size_local_y, half_size_local_z = box_local_size / 2.0
                    if (abs(voxel_center_local[0]) <= half_size_local_x and
                        abs(voxel_center_local[1]) <= half_size_local_y and
                            abs(voxel_center_local[2]) <= half_size_local_z):
                        grid[k_depth_idx, j_height_idx, i_width_idx] = 1.0
                        break
    return grid


class FrankaShelfEnv(VecEnv):
    FRANKA_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4',
                          'joint5', 'joint6', 'joint7', 'finger_joint1', 'finger_joint2']
    FRANKA_NUM_ARM_JOINTS = 7
    FRANKA_NUM_TOTAL_JOINTS = len(FRANKA_JOINT_NAMES)
    FRANKA_DEFAULT_INITIAL_QPOS = np.array(
        [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04], dtype=np.float32)

    FRANKA_QPOS_LOWER = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    FRANKA_QPOS_UPPER = np.array(
        [2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)
    FRANKA_VEL_LIMIT = np.array(
        [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float32)

    DEFAULT_PLATE_WIDTH, DEFAULT_PLATE_DEPTH, DEFAULT_PLATE_THICKNESS = 0.5, 0.4, 0.02
    DEFAULT_WALL_THICKNESS, DEFAULT_ACTUAL_OPENING_HEIGHT = 0.02, 0.25
    DEFAULT_WALL_COMPONENT_HEIGHT = DEFAULT_ACTUAL_OPENING_HEIGHT

    COMPONENT_SIZE_PLATE = np.array(
        [DEFAULT_PLATE_WIDTH, DEFAULT_PLATE_DEPTH, DEFAULT_PLATE_THICKNESS], dtype=np.float32)
    COMPONENT_SIZE_SIDE_WALL = np.array(
        [DEFAULT_WALL_THICKNESS, DEFAULT_PLATE_DEPTH, DEFAULT_WALL_COMPONENT_HEIGHT], dtype=np.float32)
    COMPONENT_SIZE_BACK_WALL = np.array(
        [DEFAULT_PLATE_WIDTH, DEFAULT_WALL_THICKNESS, DEFAULT_WALL_COMPONENT_HEIGHT], dtype=np.float32)
    FIXED_COMPONENT_SIZES = [COMPONENT_SIZE_PLATE, COMPONENT_SIZE_PLATE,
                             COMPONENT_SIZE_SIDE_WALL, COMPONENT_SIZE_SIDE_WALL, COMPONENT_SIZE_BACK_WALL]
    SHELF_NUM_COMPONENTS = len(FIXED_COMPONENT_SIZES)

    def __init__(self, render_mode: Optional[str] = None,
                 num_envs: int = 1,
                 env_spacing: Tuple[float, float] = (2.0, 2.0),
                 voxel_grid_dims: Tuple[int, int, int] = (16, 16, 16),
                 workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
                     (-0.8, 0.8), (-0.8, 0.8), (0.0, 1.5)),
                 voxel_grid_world_size: Tuple[float,
                                              float, float] = (1.6, 1.6, 1.0),
                 max_steps_per_episode: int = 1000,
                 dt: float = 0.01,
                 franka_xml_path: str = 'xml/franka_emika_panda/panda.xml',
                 k_dist_reward: float = 1.0, k_time_penalty: float = 0.01, k_action_penalty: float = 0.001,
                 k_joint_limit_penalty: float = 10.0, k_collision_penalty: float = 100.0,
                 success_reward_val: float = 200.0, success_threshold_val: float = 0.05,
                 k_accel_penalty: float = 0.0,
                 video_camera_pos: Tuple[float,
                                         float, float] = (1.8, -1.8, 2.0),
                 video_camera_lookat: Tuple[float,
                                            float, float] = (0.3, 0.0, 0.5),
                 video_camera_fov: float = 45,
                 video_res: Tuple[int, int] = (320, 240)
                 ):
        self.render_mode = render_mode

        self._num_envs = num_envs  # Used by super() if it needs num_envs from self

        # Define observation and action spaces
        robot_state_flat_dim = self.FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4
        relative_target_pos_dim = 3
        _observation_space = spaces.Dict({
            "robot_state_flat": spaces.Box(low=np.full(robot_state_flat_dim, -np.inf, dtype=np.float32), high=np.full(robot_state_flat_dim, np.inf, dtype=np.float32), dtype=np.float32),
            "relative_target_pos": spaces.Box(low=np.full(relative_target_pos_dim, -np.inf, dtype=np.float32), high=np.full(relative_target_pos_dim, np.inf, dtype=np.float32), dtype=np.float32),
            "obstacle_voxels": spaces.Box(low=0, high=1, shape=voxel_grid_dims, dtype=np.float32)
        })
        _action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.FRANKA_NUM_ARM_JOINTS,), dtype=np.float32)

        super().__init__(num_envs=self._num_envs,
                         observation_space=_observation_space, action_space=_action_space)

        # Initialize other attributes AFTER super().__init__()
        self.env_spacing = env_spacing
        self.voxel_grid_dims = voxel_grid_dims
        self.workspace_bounds_x, self.workspace_bounds_y, self.workspace_bounds_z = workspace_bounds
        self.voxel_grid_world_span_x, self.voxel_grid_world_span_y, self.voxel_grid_world_span_z = voxel_grid_world_size

        _vx, _vy, _vz = self.voxel_grid_world_span_x / self.voxel_grid_dims[2], \
            self.voxel_grid_world_span_y / self.voxel_grid_dims[1], \
            self.voxel_grid_world_span_z / self.voxel_grid_dims[0]
        self.voxel_size = min(_vx, _vy, _vz)
        eff_w, eff_h, eff_d = self.voxel_grid_dims[2]*self.voxel_size, \
            self.voxel_grid_dims[1]*self.voxel_size, \
            self.voxel_grid_dims[0]*self.voxel_size

        self.grid_origin_x = self.workspace_bounds_x[0] + (
            (self.workspace_bounds_x[1]-self.workspace_bounds_x[0]) - eff_w)/2.0
        self.grid_origin_y = self.workspace_bounds_y[0] + (
            (self.workspace_bounds_y[1]-self.workspace_bounds_y[0]) - eff_h)/2.0
        self.grid_origin_z = self.workspace_bounds_z[0]
        self.grid_origin_global = np.array(
            [self.grid_origin_x, self.grid_origin_y, self.grid_origin_z], dtype=np.float32)
        self.grid_origins_batch = np.tile(
            self.grid_origin_global, (self.num_envs, 1))

        self.max_steps = max_steps_per_episode
        self.current_step = np.zeros(self.num_envs, dtype=np.int32)
        self.dt = dt
        self.franka_xml_path = franka_xml_path

        self.k_dist_reward, self.k_time_penalty, self.k_action_penalty = k_dist_reward, k_time_penalty, k_action_penalty
        self.k_joint_limit_penalty, self.k_collision_penalty = k_joint_limit_penalty, k_collision_penalty
        self.success_reward, self.success_threshold = success_reward_val, success_threshold_val
        self.k_accel_penalty = k_accel_penalty

        self._define_shelf_configurations()
        self.current_shelf_config_key = [""] * self.num_envs
        self.current_shelf_instance_params = [{} for _ in range(self.num_envs)]

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
            show_viewer=self.render_mode == "human"  # type: ignore
        )

        self.video_camera_params = {
            'pos': video_camera_pos, 'lookat': video_camera_lookat,
            'fov': video_camera_fov, 'res': video_res
        }
        self._is_recording = False
        # print("Initializing video camera in FrankaShelfEnv __init__...") # For debugging
        self.video_camera = self.scene.add_camera(
            res=self.video_camera_params['res'],
            pos=self.video_camera_params['pos'],
            lookat=self.video_camera_params['lookat'],
            fov=self.video_camera_params['fov'],
            GUI=False
        )
        # if self.video_camera: print(f"Video camera added to scene in __init__.") # For debugging
        # else: print("Warning: Failed to add video camera to scene in __init__.") # For debugging

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=self.franka_xml_path))
        self.shelf_component_entities = [
            self.scene.add_entity(gs.morphs.Box(pos=(0, -10-i*0.5, 0), quat=(1, 0, 0, 0), size=tuple(
                s), fixed=True, visualization=True, collision=True))  # type: ignore
            for i, s in enumerate(self.FIXED_COMPONENT_SIZES)
        ]

        if not self.scene.is_built:
            # print(f"Building Genesis scene for {self.num_envs} internal environments with spacing {self.env_spacing}...") # For debugging
            self.scene.build(n_envs=self.num_envs,
                             env_spacing=self.env_spacing)
            # print("Scene built.") # For debugging
        # else: print("Warning: Scene was already built before FrankaShelfEnv tried to build it.") # For debugging

        self.target_position_world = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        self.current_voxel_grid = np.zeros(
            (self.num_envs,) + self.voxel_grid_dims, dtype=np.float32)
        self.prev_joint_vel = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)

        self.franka_dof_indices = np.array([self.franka.get_joint(
            name).dof_idx_local for name in self.FRANKA_JOINT_NAMES], dtype=np.int32)
        self.franka_arm_dof_indices = self.franka_dof_indices[:self.FRANKA_NUM_ARM_JOINTS]
        self.robot_ee_link_name = "hand"
        self.max_joint_velocity_scale = 0.5
        self.actions = np.zeros(
            (self.num_envs, self.action_space.shape[0]), dtype=self.action_space.dtype)
        self.buf_infos: List[Dict[str, Any]] = [{}
                                                for _ in range(self.num_envs)]

    def _define_shelf_configurations(self):
        self.shelf_configurations = {
            "high_center_reach": {"name": "high_center_reach", "base_pos_range_x": (0.45, 0.65), "base_pos_range_y": (-0.2, 0.2), "base_pos_range_z": (0.5, 0.65)},
            "low_forward_reach": {"name": "low_forward_reach", "base_pos_range_x": (0.40, 0.60), "base_pos_range_y": (-0.2, 0.2), "base_pos_range_z": (0.05, 0.20)},
            "mid_side_reach_right": {"name": "mid_side_reach_right", "base_pos_range_x": (0.25, 0.5), "base_pos_range_y": (0.4, 0.6), "base_pos_range_z": (0.25, 0.45)},
            "mid_side_reach_left": {"name": "mid_side_reach_left", "base_pos_range_x": (0.25, 0.5), "base_pos_range_y": (-0.6, -0.4), "base_pos_range_z": (0.25, 0.45)}
        }
        self.shelf_config_keys = list(self.shelf_configurations.keys())

    def _get_robot_state_parts_batched(self):
        joint_pos_all = to_numpy(
            self.franka.get_dofs_position(self.franka_dof_indices))
        joint_vel_all = to_numpy(
            self.franka.get_dofs_velocity(self.franka_dof_indices))
        arm_joint_pos = joint_pos_all[:, :self.FRANKA_NUM_ARM_JOINTS]
        arm_joint_vel = joint_vel_all[:, :self.FRANKA_NUM_ARM_JOINTS]
        ee_link = self.franka.get_link(self.robot_ee_link_name)
        ee_pos = to_numpy(ee_link.get_pos())
        ee_orient_quat_wxyz = to_numpy(ee_link.get_quat())
        return arm_joint_pos, arm_joint_vel, ee_pos, ee_orient_quat_wxyz

    def _build_shelf_structure_batched(self):
        shelf_assembly_origins_world = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        shelf_assembly_yaws = np.zeros(self.num_envs, dtype=np.float32)
        for i in range(self.num_envs):
            config_key = random.choice(self.shelf_config_keys)
            self.current_shelf_config_key[i] = config_key
            config = self.shelf_configurations[config_key]
            shelf_assembly_origins_world[i, 0] = random.uniform(
                *config["base_pos_range_x"])
            shelf_assembly_origins_world[i, 1] = random.uniform(
                *config["base_pos_range_y"])
            shelf_assembly_origins_world[i, 2] = random.uniform(
                *config["base_pos_range_z"])
            if "center" in config["name"] or "forward" in config["name"]:
                shelf_assembly_yaws[i] = math.atan2(
                    -shelf_assembly_origins_world[i, 1], -shelf_assembly_origins_world[i, 0]) - np.pi / 2.0
            elif "right" in config["name"]:
                shelf_assembly_yaws[i] = np.pi
            elif "left" in config["name"]:
                shelf_assembly_yaws[i] = 0.0
            else:
                shelf_assembly_yaws[i] = 0.0
            self.current_shelf_instance_params[i] = {
                "shelf_assembly_origin_world": shelf_assembly_origins_world[i].copy(),
                "shelf_assembly_yaw": shelf_assembly_yaws[i],
                "actual_opening_height": self.DEFAULT_ACTUAL_OPENING_HEIGHT,
                "internal_width": self.DEFAULT_PLATE_WIDTH - 2*self.DEFAULT_WALL_THICKNESS,
                "internal_depth": self.DEFAULT_PLATE_DEPTH - self.DEFAULT_WALL_THICKNESS,
            }
        cos_yaw_half = np.cos(shelf_assembly_yaws / 2.0)
        sin_yaw_half = np.sin(shelf_assembly_yaws / 2.0)
        shelf_assembly_world_quats_batch = np.zeros(
            (self.num_envs, 4), dtype=np.float32)
        shelf_assembly_world_quats_batch[:, 0] = cos_yaw_half
        shelf_assembly_world_quats_batch[:, 3] = sin_yaw_half
        bottom_plate_s, top_plate_s, left_wall_s, _, back_wall_s = self.FIXED_COMPONENT_SIZES[
            0:5]
        actual_opening_h = self.DEFAULT_ACTUAL_OPENING_HEIGHT
        local_offsets = [
            np.array([0, 0, bottom_plate_s[2]/2.0]),
            np.array([0, 0, bottom_plate_s[2] +
                     actual_opening_h+top_plate_s[2]/2.0]),
            np.array([-(bottom_plate_s[0]/2.0-left_wall_s[0]/2.0),
                     0, bottom_plate_s[2]+left_wall_s[2]/2.0]),
            np.array([(bottom_plate_s[0]/2.0-left_wall_s[0]/2.0),
                     0, bottom_plate_s[2]+left_wall_s[2]/2.0]),
            np.array([0, -(bottom_plate_s[1]/2.0-back_wall_s[1]/2.0),
                     bottom_plate_s[2]+back_wall_s[2]/2.0])
        ]
        list_of_box_params_for_voxelization_batched = []
        for comp_idx in range(self.SHELF_NUM_COMPONENTS):
            entity = self.shelf_component_entities[comp_idx]
            local_offset_single = local_offsets[comp_idx]
            fixed_size_single = self.FIXED_COMPONENT_SIZES[comp_idx]
            batched_final_comp_pos_world = np.zeros(
                (self.num_envs, 3), dtype=np.float32)
            for env_i in range(self.num_envs):
                R_shelf_assembly_yaw_i = quat_to_rotation_matrix(
                    shelf_assembly_world_quats_batch[env_i])
                rotated_offset_world_i = R_shelf_assembly_yaw_i @ local_offset_single
                batched_final_comp_pos_world[env_i] = shelf_assembly_origins_world[env_i] + \
                    rotated_offset_world_i
            pos_tensor = torch.tensor(
                batched_final_comp_pos_world, device=gs.device, dtype=torch.float32)
            quat_tensor = torch.tensor(
                shelf_assembly_world_quats_batch, device=gs.device, dtype=torch.float32)
            entity.set_pos(pos_tensor)
            entity.set_quat(quat_tensor)
            list_of_box_params_for_voxelization_batched.append(
                {'position': batched_final_comp_pos_world, 'size': fixed_size_single, 'orientation_quat': shelf_assembly_world_quats_batch})
        return list_of_box_params_for_voxelization_batched

    def _generate_target_in_shelf_batched(self):
        self.target_position_world = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        for i in range(self.num_envs):
            params = self.current_shelf_instance_params[i]
            open_w, open_d, open_h = params["internal_width"], params["internal_depth"], params["actual_opening_height"]
            target_local = np.array([random.uniform(-open_w/2*.6, open_w/2*.6), random.uniform(-open_d /
                                    2*.6, open_d/2*.2), random.uniform(-open_h/2*.7, open_h/2*.7)], dtype=np.float32)
            shelf_origin = params["shelf_assembly_origin_world"]
            offset_to_opening_center_local = np.array(
                [0, 0, self.DEFAULT_PLATE_THICKNESS+open_h/2.0], dtype=np.float32)
            yaw_i = params["shelf_assembly_yaw"]
            quat_i = np.array(
                [np.cos(yaw_i/2.0), 0, 0, np.sin(yaw_i/2.0)], dtype=np.float32)
            R_shelf_yaw_i = quat_to_rotation_matrix(quat_i)
            offset_world_rotated = R_shelf_yaw_i @ offset_to_opening_center_local
            opening_center_world = shelf_origin + offset_world_rotated
            target_offset_world_rotated = R_shelf_yaw_i @ target_local
            self.target_position_world[i] = opening_center_world + \
                target_offset_world_rotated
        if self.render_mode == "human" or self._is_recording:  # type: ignore
            if hasattr(self.scene, 'clear_debug_objects'):
                self.scene.clear_debug_objects()
            if self.num_envs == 1:
                self.scene.draw_debug_spheres(
                    self.target_position_world.tolist(), radius=0.03, color=(0, 1, 0, 0.8))
        return self.target_position_world

    def _get_obs_batched(self) -> VecEnvObs:
        arm_joint_pos, arm_joint_vel, ee_pos, ee_orient_quat_wxyz = self._get_robot_state_parts_batched()
        robot_state_flat = np.concatenate(
            [arm_joint_pos, arm_joint_vel, ee_pos, ee_orient_quat_wxyz], axis=1).astype(np.float32)
        relative_target_pos = (
            self.target_position_world - ee_pos).astype(np.float32)
        return {"robot_state_flat": robot_state_flat,
                "relative_target_pos": relative_target_pos,
                "obstacle_voxels": self.current_voxel_grid.copy()}

    def reset(self) -> VecEnvObs:
        if self.scene.is_built:
            self.scene.reset()
        else:
            # print("Warning: reset() called but scene was not built. Rebuilding...") # For debugging
            self.scene.build(n_envs=self.num_envs,
                             env_spacing=self.env_spacing)

        self.current_step.fill(0)
        initial_qpos_batch = np.tile(
            self.FRANKA_DEFAULT_INITIAL_QPOS, (self.num_envs, 1))
        initial_qvel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_TOTAL_JOINTS), dtype=np.float32)
        qpos_tensor = torch.tensor(
            initial_qpos_batch, device=gs.device, dtype=torch.float32)
        qvel_tensor = torch.tensor(
            initial_qvel_batch, device=gs.device, dtype=torch.float32)
        self.franka.set_dofs_position(qpos_tensor, self.franka_dof_indices)
        self.franka.set_dofs_velocity(qvel_tensor, self.franka_dof_indices)
        self.prev_joint_vel.fill(0.0)
        batched_box_params = self._build_shelf_structure_batched()
        self._generate_target_in_shelf_batched()
        self.current_voxel_grid = boxes_to_voxel_grid_batched(
            batched_box_params, self.voxel_grid_dims, self.grid_origins_batch, self.voxel_size, self.num_envs)
        obs_batch = self._get_obs_batched()
        self.buf_infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.buf_infos[i]["shelf_config"] = self.current_shelf_config_key[i]
        return obs_batch

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions.copy()

    def step_wait(self) -> VecEnvStepReturn:
        actions_batch = self.actions
        self.current_step += 1
        actions_batch_clipped = np.clip(
            actions_batch, -1.0, 1.0).astype(np.float32)
        scaled_actions_batch = actions_batch_clipped * self.max_joint_velocity_scale
        actions_tensor = torch.tensor(
            scaled_actions_batch, device=gs.device, dtype=torch.float32)
        self.franka.control_dofs_velocity(
            actions_tensor, self.franka_arm_dof_indices)
        self.scene.step()

        if self._is_recording and self.video_camera:
            self.video_camera.render()

        arm_joint_pos_b, arm_joint_vel_b, ee_pos_b, _ = self._get_robot_state_parts_batched()
        dist_to_target_b = np.linalg.norm(
            ee_pos_b - self.target_position_world, axis=1)
        reward_distance_b = -self.k_dist_reward * dist_to_target_b
        time_penalty_b = -self.k_time_penalty * \
            np.ones(self.num_envs, dtype=np.float32)
        action_penalty_b = -self.k_action_penalty * \
            np.sum(np.square(actions_batch_clipped), axis=1)
        joint_pos_penalty_b = np.zeros(self.num_envs, dtype=np.float32)
        near_limit_thresh = 0.05
        for i in range(self.FRANKA_NUM_ARM_JOINTS):
            lower_limit_dist = arm_joint_pos_b[:, i] - \
                (self.FRANKA_QPOS_LOWER[i] + near_limit_thresh)
            upper_limit_dist = (
                self.FRANKA_QPOS_UPPER[i] - near_limit_thresh) - arm_joint_pos_b[:, i]
            joint_pos_penalty_b -= self.k_joint_limit_penalty * \
                np.maximum(0, -lower_limit_dist)
            joint_pos_penalty_b -= self.k_joint_limit_penalty * \
                np.maximum(0, -upper_limit_dist)

        collision_penalty_val_b = np.zeros(self.num_envs, dtype=np.float32)
        terminated_collision_b = np.zeros(self.num_envs, dtype=bool)
        if self.scene.is_built:
            contacts_data = self.franka.get_contacts()
            if contacts_data and 'valid_mask' in contacts_data and \
                    'link_a' in contacts_data and 'link_b' in contacts_data and \
                    hasattr(self.scene, 'rigid_solver') and hasattr(self.scene.rigid_solver, 'links'):

                # Shape: (n_envs, n_max_contacts)
                valid_mask_np = to_numpy(contacts_data['valid_mask'])
                # Iterate through each environment
                for i_env in range(self.num_envs):
                    env_had_collision_this_step = False  # Flag for the current environment

                    # Iterate over max_contacts slots
                    for i_contact in range(valid_mask_np.shape[1]):
                        # This contact is valid for i_env
                        if valid_mask_np[i_env, i_contact]:
                            env_had_collision_this_step = True

                    # After checking all contacts for i_env:
                    if env_had_collision_this_step:
                        collision_penalty_val_b[i_env] -= self.k_collision_penalty
                        terminated_collision_b[i_env] = True

        success_reward_val_b = np.zeros(self.num_envs, dtype=np.float32)
        terminated_success_b = dist_to_target_b < self.success_threshold
        success_reward_val_b[terminated_success_b] = self.success_reward

        accel_penalty_b = np.zeros(self.num_envs, dtype=np.float32)
        if self.k_accel_penalty > 0 and self.dt > 0:
            joint_accel_b = (arm_joint_vel_b - self.prev_joint_vel) / self.dt
            accel_penalty_b = -self.k_accel_penalty * \
                np.sum(np.square(joint_accel_b), axis=1)

        rewards_b = reward_distance_b + time_penalty_b + action_penalty_b + \
            joint_pos_penalty_b + collision_penalty_val_b + success_reward_val_b + \
            accel_penalty_b
        terminated_b = np.logical_or(
            terminated_collision_b, terminated_success_b)
        truncated_b = self.current_step >= self.max_steps
        dones_b = np.logical_or(terminated_b, truncated_b)
        observations_b = self._get_obs_batched()

        self.buf_infos = []
        for i in range(self.num_envs):
            info_dict = {"is_success": terminated_success_b[i],
                         "distance_to_target": float(dist_to_target_b[i]),
                         "collision_detected": terminated_collision_b[i],
                         "current_step": self.current_step[i],
                         "shelf_config": self.current_shelf_config_key[i]}
            if dones_b[i]:
                info_dict["terminal_observation"] = {
                    k: v[i] for k, v in observations_b.items()}
                if truncated_b[i] and not terminated_b[i]:
                    info_dict["TimeLimit.truncated"] = True
            self.buf_infos.append(info_dict)
        self.prev_joint_vel = arm_joint_vel_b.copy()
        return observations_b, rewards_b, dones_b, self.buf_infos

    def render(self, mode: Optional[str] = 'human'):  # type: ignore
        if mode == "human" and self.render_mode == "human":  # type: ignore
            return None
        elif mode == "rgb_array" and self.num_envs > 0:
            if not self.video_camera:
                # print("Warning: video_camera not found in render('rgb_array'). Should be pre-initialized.") # For debugging
                grid_to_render = self.current_voxel_grid[0]
                img_slice = np.sum(grid_to_render, axis=0)
                img_slice_norm = np.clip(img_slice, 0, np.max(
                    img_slice) if np.max(img_slice) > 0 else 1.0)
                if np.max(img_slice_norm) > 0:
                    img_slice_norm = img_slice_norm / np.max(img_slice_norm)
                img_slice_uint8 = (img_slice_norm * 255).astype(np.uint8)
                rgb_image = np.stack([img_slice_uint8]*3, axis=-1)
                return [rgb_image for _ in range(self.num_envs)] if self.num_envs > 1 else rgb_image

            if self.video_camera:
                rgb, _, _, _ = self.video_camera.render(
                    rgb=True, depth=False, segmentation=False, normal=False)
                if isinstance(rgb, list) and len(rgb) == self.num_envs:
                    return rgb[0]
                return rgb
        return None

    def close(self) -> None:
        if self.video_camera and hasattr(self.video_camera, 'close'):
            self.video_camera.close()
        if hasattr(self, 'scene') and self.scene is not None:
            if hasattr(self.scene, 'close') and callable(self.scene.close):
                self.scene.close()
            elif hasattr(self.scene, 'shutdown') and callable(self.scene.shutdown):
                self.scene.shutdown()
        # print("FrankaShelfEnv (VecEnv) closed.") # For debugging

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_obj = self
        if hasattr(target_obj, attr_name):
            data = getattr(target_obj, attr_name)
            if indices is None:
                return [data for _ in range(self.num_envs)]
            return [data for i in self._get_indices(indices)]
        else:
            raise AttributeError(f"Attribute {attr_name} not found")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        data = []
        for i in self._get_indices(indices):
            if hasattr(self, method_name):
                data.append(getattr(self, method_name)(
                    *method_args, **method_kwargs))
            else:
                raise AttributeError(f"Method {method_name} not found")
        return data

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        if indices is None:
            return [isinstance(self, wrapper_class) for _ in range(self.num_envs)]
        return [isinstance(self, wrapper_class) for i in self._get_indices(indices)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:  # type: ignore
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed for _ in range(self.num_envs)]  # type: ignore

    def setup_video_camera_if_needed(self, env_idx_to_focus: int = 0):
        if self.video_camera is None:
            # print("CRITICAL WARNING: video_camera was None in setup_video_camera_if_needed. This should not happen if __init__ is correct.") # For debugging
            self.video_camera = self.scene.add_camera(
                res=self.video_camera_params['res'],
                pos=self.video_camera_params['pos'],
                lookat=self.video_camera_params['lookat'],
                fov=self.video_camera_params['fov'],
                GUI=False,
                env_idx=env_idx_to_focus
            )
            # if self.video_camera: print(f"Video camera emergency re-added to scene (focused on env_idx {env_idx_to_focus}).") # For debugging
            # else: print(f"Failed to emergency re-add video camera.") # For debugging

    def start_video_recording(self, env_idx_to_focus: int = 0):
        if self.video_camera:
            try:
                self.video_camera.start_recording()
                self._is_recording = True
                if self._is_recording and hasattr(self.scene, 'draw_debug_spheres'):
                    if hasattr(self.scene, 'clear_debug_objects'):
                        self.scene.clear_debug_objects()
                    if self.num_envs == 1:
                        self.scene.draw_debug_spheres(
                            self.target_position_world.tolist(), radius=0.03, color=(0, 1, 0, 0.8))

                # print(f"Video recording started (env_idx {env_idx_to_focus} perspective).") # For debugging
            except Exception as e:
                # print(f"Error starting video recording: {e}") # For debugging
                self._is_recording = False
        else:
            # print("Cannot start recording: Video camera not available (should be initialized in __init__).") # For debugging
            self._is_recording = False
        return self._is_recording

    def stop_video_recording(self, save_dir: str, filename: str, fps: int = 30) -> Optional[str]:
        if self.video_camera and self._is_recording:
            try:
                os.makedirs(save_dir, exist_ok=True)
                full_path = os.path.join(save_dir, filename)
                self.video_camera.stop_recording(
                    save_to_filename=full_path, fps=fps)
                self._is_recording = False
                # print(f"Video recording stopped. Saved to: {full_path}") # For debugging
                return full_path
            except Exception as e:
                # print(f"Error stopping/saving video recording: {e}") # For debugging
                self._is_recording = False
        elif not self._is_recording:
            # print("Not stopping recording as it wasn't active.") # For debugging
            pass
        # else: print("Cannot stop recording: Video camera not available or not recording.") # For debugging
        return None

    def get_scene_and_camera(self) -> Tuple[Optional[Any], Optional[Any]]:
        return self.scene, self.video_camera


if __name__ == '__main__':
    # print("Attempting to initialize Genesis...") # For debugging
    try:
        if not hasattr(gs, '_is_initialized') or not gs._is_initialized:
            gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
        # print("Genesis initialized successfully.") # For debugging
    except Exception as e:
        # print(f"CRITICAL ERROR: Failed to initialize Genesis: {e}\nExiting.") # For debugging
        exit()

    NUM_GENESIS_INTERNAL_ENVS = 2
    # print(f"Attempting to create FrankaShelfEnv as VecEnv with num_envs={NUM_GENESIS_INTERNAL_ENVS}...") # For debugging
    env = None
    try:
        env = FrankaShelfEnv(
            num_envs=NUM_GENESIS_INTERNAL_ENVS,
            render_mode=None,
            k_collision_penalty=150.0,
            k_accel_penalty=0.001,
            video_res=(160, 120)
        )
        # print(f"FrankaShelfEnv (VecEnv) created successfully with {env.num_envs} environments.") # For debugging
        # print(f"Observation Space: {env.observation_space}") # For debugging
        # print(f"Action Space: {env.action_space}") # For debugging

        # print("\nTesting VecEnv reset...") # For debugging
        obs_batch = env.reset()
        # print(f"Reset successful. Obs batch 'obstacle_voxels' shape: {obs_batch['obstacle_voxels'].shape}") # For debugging
        assert obs_batch['obstacle_voxels'].shape[0] == NUM_GENESIS_INTERNAL_ENVS

        # print("\nTesting video recording...") # For debugging
        if env.start_video_recording(env_idx_to_focus=0):
            # print("Recording started for 50 steps...") # For debugging
            for step_i in range(50):
                actions = np.array([env.action_space.sample()
                                   for _ in range(env.num_envs)])
                env.step_async(actions)
                _, _, dones, _ = env.step_wait()
                # if (step_i + 1) % 10 == 0: print(f"  Video test step {step_i+1}") # For debugging
                if np.any(dones):
                    # print("  Resetted envs during video test.") # For debugging
                    env.reset()
            video_path = env.stop_video_recording(
                save_dir="./videos_test", filename="test_franka_env.mp4", fps=15)
            # if video_path: print(f"Test video saved to {video_path}") # For debugging
            # else: print("Test video saving failed.") # For debugging
        # else: print("Could not start video recording for test.") # For debugging

        num_episodes_to_run = 1
        # print(f"\nRunning {num_episodes_to_run} example episodes...") # For debugging
        for episode_num in range(num_episodes_to_run):
            # print(f"\n--- Starting Episode {episode_num + 1} (Batch of {env.num_envs} envs) ---") # For debugging
            current_obs_batch = env.reset()
            episode_rewards = np.zeros(env.num_envs)
            max_steps_this_episode = 200
            for step_num in range(max_steps_this_episode):
                actions = np.array([env.action_space.sample()
                                   for _ in range(env.num_envs)])
                env.step_async(actions)
                next_obs_batch, rewards_b, dones_b, infos_b = env.step_wait()
                episode_rewards += rewards_b
                # if (step_num + 1) % 10 == 0: # For debugging
                # print(f"  Step {step_num+1}:") # For debugging
                # for i in range(env.num_envs): # For debugging
                # print(f"    Env {i}: Rew={rewards_b[i]:.2f}, Done={dones_b[i]}, Success={infos_b[i].get('is_success', 'N/A')}") # For debugging
                current_obs_batch = next_obs_batch
                if np.all(dones_b):
                    # print(f"  All environments finished at step {step_num+1}.") # For debugging
                    break
            # print(f"  Episode finished. Total rewards: {episode_rewards}") # For debugging
            if env.render_mode == "human":
                # print("  Pausing for 1s...") # For debugging
                import time
                time.sleep(1)

    except Exception as e:
        import traceback
        # print(f"ERROR during FrankaShelfEnv (VecEnv) example usage: {e}") # For debugging
        traceback.print_exc()
    finally:
        if env:
            # print("\nClosing FrankaShelfEnv (VecEnv)...") # For debugging
            env.close()
        if hasattr(gs, 'shutdown') and callable(gs.shutdown):
            # print("Shutting down Genesis...") # For debugging
            gs.shutdown()
        # print("Example application finished.") # For debugging
