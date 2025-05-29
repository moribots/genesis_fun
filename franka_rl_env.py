"""
Defines the FrankaShelfEnv, a vectorized reinforcement learning environment
for a Franka Emika Panda robot interacting with a shelf.

The environment involves tasks like reaching specific poses within or around a
dynamically configured shelf. The shelf is composed of 5 fixed-size box segments.
The observation space includes robot state, target position, and the explicit
parameters (pose, size) of these 5 shelf components.
The action space is continuous joint velocity control.
It uses the Genesis simulation engine for physics and rendering.
"""
import os
import sys
import random
import math
from typing import List, Any, Optional, Union, Type, Dict, Tuple
import traceback

import gymnasium as gym
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
    If the input is a PyTorch Tensor on a CUDA device, it's moved to the CPU.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy() if data.is_cuda else data.numpy()
    return np.asarray(data, dtype=np.float32)

# --- FrankaShelfEnv Class ---


class FrankaShelfEnv(VecEnv):
    """
    A vectorized environment for a Franka Emika Panda robot interacting with a shelf
    composed of 5 fixed-component boxes. Observations are robot state, target,
    and parameters of these 5 shelf components.
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
    NUM_PARAMS_PER_SHELF_COMPONENT = 3 + 4 + 3  # pos, quat_wxyz, size

    def __init__(self,
                 render_mode: Optional[str] = None,
                 num_envs: int = 1,
                 env_spacing: Tuple[float, float] = (1.5, 1.5),
                 workspace_bounds_xyz: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
                     (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.5)),
                 max_steps_per_episode: int = 1000,
                 dt: float = 0.01,
                 franka_xml_path: str = 'xml/franka_emika_panda/panda.xml',
                 k_dist_reward: float = 1.0, k_time_penalty: float = 0.01, k_action_penalty: float = 0.001,
                 k_joint_limit_penalty: float = 10.0, k_collision_penalty: float = 100.0,
                 k_accel_penalty: float = 0.0, success_reward_val: float = 200.0,
                 success_threshold_val: float = 0.05,
                 video_camera_pos: Tuple[float,
                                         float, float] = (1.8, -1.8, 2.0),
                 video_camera_lookat: Tuple[float,
                                            float, float] = (0.3, 0.0, 0.5),
                 video_camera_fov: float = 45, video_res: Tuple[int, int] = (960, 640),
                 include_shelf: bool = True,
                 randomize_shelf_config: bool = True
                 ):
        self.render_mode = render_mode
        self._num_envs = num_envs
        self.include_shelf = include_shelf
        self.randomize_shelf_config = randomize_shelf_config

        robot_state_flat_dim = self.FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4
        relative_target_pos_dim = 3
        shelf_component_params_flat_dim = self.SHELF_NUM_COMPONENTS * \
            self.NUM_PARAMS_PER_SHELF_COMPONENT

        _observation_space = spaces.Dict({
            "robot_state_flat": spaces.Box(low=np.full(robot_state_flat_dim, -np.inf, dtype=np.float32), high=np.full(robot_state_flat_dim, np.inf, dtype=np.float32), dtype=np.float32),
            "relative_target_pos": spaces.Box(low=np.full(relative_target_pos_dim, -np.inf, dtype=np.float32), high=np.full(relative_target_pos_dim, np.inf, dtype=np.float32), dtype=np.float32),
            "shelf_component_params": spaces.Box(low=-np.inf, high=np.inf, shape=(shelf_component_params_flat_dim,), dtype=np.float32)
        })
        _action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.FRANKA_NUM_ARM_JOINTS,), dtype=np.float32)

        super().__init__(num_envs=self._num_envs,
                         observation_space=_observation_space, action_space=_action_space)

        self.env_spacing = env_spacing
        self.workspace_bounds_x, self.workspace_bounds_y, self.workspace_bounds_z = workspace_bounds_xyz

        self.max_steps = max_steps_per_episode
        self.current_step_per_env = np.zeros(self.num_envs, dtype=np.int32)
        self.dt = dt
        self.franka_xml_path = franka_xml_path
        self.k_dist_reward, self.k_time_penalty, self.k_action_penalty = k_dist_reward, k_time_penalty, k_action_penalty
        self.k_joint_limit_penalty, self.k_collision_penalty, self.k_accel_penalty = k_joint_limit_penalty, k_collision_penalty, k_accel_penalty
        self.success_reward, self.success_threshold = success_reward_val, success_threshold_val

        self._define_shelf_configurations()
        self.current_shelf_config_key_per_env: List[str] = [""] * self.num_envs
        self.current_shelf_instance_params_per_env: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)]

        self.metadata = {'render_modes': [
            'human', 'rgb_array', 'video'], 'render_fps': 60}
        sim_viewer_options = gs.options.ViewerOptions(camera_pos=video_camera_pos, camera_lookat=video_camera_lookat, camera_fov=video_camera_fov,
                                                      res=video_res, max_FPS=self.metadata['render_fps'])
        self.scene = gs.Scene(viewer_options=sim_viewer_options, sim_options=gs.options.SimOptions(
            dt=self.dt), show_viewer=(self.render_mode == "human"))
        self.video_camera_params = {
            'pos': video_camera_pos, 'lookat': video_camera_lookat, 'fov': video_camera_fov, 'res': video_res}
        self._is_recording_active = False
        self.video_capture_camera = self.scene.add_camera(
            res=self.video_camera_params['res'], pos=self.video_camera_params['pos'], lookat=self.video_camera_params['lookat'], fov=self.video_camera_params['fov'], GUI=False)
        self.plane_entity = self.scene.add_entity(gs.morphs.Plane())
        self.franka_entity = self.scene.add_entity(
            gs.morphs.MJCF(file=self.franka_xml_path))

        self.shelf_component_entities: List[Any] = []
        if self.include_shelf:
            self.shelf_component_entities = [self.scene.add_entity(gs.morphs.Box(pos=(0, -10 - i * 0.5, 0), quat=(
                1, 0, 0, 0), size=tuple(s), fixed=True, collision=True, visualization=True)) for i, s in enumerate(self.FIXED_COMPONENT_SIZES)]

        self.target_sphere_entity_type: Optional[Any] = None
        if self.scene and self._num_envs > 1:
            self.target_sphere_entity_type = self.scene.add_entity(
                gs.morphs.Sphere(
                    pos=(0, -20, 0),
                    radius=0.03,
                    visualization=True,
                    collision=False,
                    fixed=True
                )
            )

        if not self.scene.is_built:
            self.scene.build(n_envs=self.num_envs,
                             env_spacing=self.env_spacing)

        self.shelf_component_params_batch = np.zeros(
            (self.num_envs, self.SHELF_NUM_COMPONENTS, self.NUM_PARAMS_PER_SHELF_COMPONENT), dtype=np.float32)
        self.target_position_world_batch = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        self.prev_joint_vel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)
        self.franka_all_dof_indices_local = np.array([self.franka_entity.get_joint(
            name).dof_idx_local for name in self.FRANKA_JOINT_NAMES], dtype=np.int32)
        self.franka_arm_dof_indices_local = self.franka_all_dof_indices_local[
            :self.FRANKA_NUM_ARM_JOINTS]
        self.max_allowable_joint_velocity_scale = 0.5
        self.actions_buffer = np.zeros(
            (self.num_envs, self.action_space.shape[0]), dtype=self.action_space.dtype)
        self.buf_infos: List[Dict[str, Any]] = [{}
                                                for _ in range(self.num_envs)]

    def _define_shelf_configurations(self) -> None:
        self.shelf_configurations: Dict[str, Dict[str, Any]] = {
            "default_center_reach": {"name": "default_center_reach", "base_pos_range_x": (0.55, 0.55), "base_pos_range_y": (0.0, 0.0), "base_pos_range_z": (0.3, 0.3)},
            "high_center_reach": {"name": "high_center_reach", "base_pos_range_x": (0.45, 0.65), "base_pos_range_y": (-0.2, 0.2), "base_pos_range_z": (0.5, 0.65)},
            "low_forward_reach": {"name": "low_forward_reach", "base_pos_range_x": (0.40, 0.60), "base_pos_range_y": (-0.2, 0.2), "base_pos_range_z": (0.05, 0.20)},
            "mid_side_reach_right": {"name": "mid_side_reach_right", "base_pos_range_x": (0.25, 0.5), "base_pos_range_y": (0.4, 0.6), "base_pos_range_z": (0.25, 0.45)},
            "mid_side_reach_left": {"name": "mid_side_reach_left", "base_pos_range_x": (0.25, 0.5), "base_pos_range_y": (-0.6, -0.4), "base_pos_range_z": (0.25, 0.45)}
        }
        self.shelf_config_keys_list: List[str] = list(
            self.shelf_configurations.keys())
        self.default_shelf_config_key: str = "default_center_reach"

    def _get_robot_state_parts_batched(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def _build_shelf_structure_and_populate_params_batched(self) -> None:
        self.shelf_component_params_batch.fill(0.0)

        shelf_assembly_origins_world_batch = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        shelf_assembly_yaws_batch = np.zeros(self.num_envs, dtype=np.float32)

        for i in range(self.num_envs):
            if self.randomize_shelf_config:
                config_key = random.choice(self.shelf_config_keys_list)
            else:
                config_key = self.default_shelf_config_key

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

        bottom_plate_s, top_plate_s, side_wall_s, _, back_wall_s = self.FIXED_COMPONENT_SIZES[
            0:5]
        actual_opening_h = self.DEFAULT_ACTUAL_OPENING_HEIGHT
        local_component_offsets = [
            np.array([0, 0, bottom_plate_s[2]/2.0]), np.array([0, 0,
                                                               bottom_plate_s[2]+actual_opening_h+top_plate_s[2]/2.0]),
            np.array([-(bottom_plate_s[0]/2.0-side_wall_s[0]/2.0),
                     0, bottom_plate_s[2]+side_wall_s[2]/2.0]),
            np.array([(bottom_plate_s[0]/2.0-side_wall_s[0]/2.0),
                     0, bottom_plate_s[2]+side_wall_s[2]/2.0]),
            np.array([0, -(bottom_plate_s[1]/2.0-back_wall_s[1]/2.0),
                     bottom_plate_s[2]+back_wall_s[2]/2.0])
        ]

        for comp_idx, component_entity in enumerate(self.shelf_component_entities):
            fixed_size_of_this_component = self.FIXED_COMPONENT_SIZES[comp_idx]
            batched_final_comp_pos_world = np.zeros(
                (self.num_envs, 3), dtype=np.float32)

            for env_i in range(self.num_envs):
                assembly_quat_xyzw_i = shelf_assembly_world_quats_batch_wxyz[env_i, [
                    1, 2, 3, 0]]
                R_shelf_assembly_yaw_i = ScipyRotation.from_quat(
                    assembly_quat_xyzw_i).as_matrix()
                rotated_offset_world_i = R_shelf_assembly_yaw_i @ local_component_offsets[comp_idx]
                batched_final_comp_pos_world[env_i] = shelf_assembly_origins_world_batch[env_i] + \
                    rotated_offset_world_i

                self.shelf_component_params_batch[env_i, comp_idx,
                                                  0:3] = batched_final_comp_pos_world[env_i]
                self.shelf_component_params_batch[env_i, comp_idx,
                                                  3:7] = shelf_assembly_world_quats_batch_wxyz[env_i]
                self.shelf_component_params_batch[env_i,
                                                  comp_idx, 7:10] = fixed_size_of_this_component

            pos_tensor = torch.tensor(
                batched_final_comp_pos_world, device=gs.device, dtype=torch.float32)
            quat_tensor = torch.tensor(
                shelf_assembly_world_quats_batch_wxyz, device=gs.device, dtype=torch.float32)
            component_entity.set_pos(pos_tensor)
            component_entity.set_quat(quat_tensor)

    def _draw_workspace_bounds(self) -> None:
        """Draws the workspace boundaries as a semi-transparent box."""
        if not self.scene:
            return
        min_coords = [self.workspace_bounds_x[0],
                      self.workspace_bounds_y[0], self.workspace_bounds_z[0]]
        max_coords = [self.workspace_bounds_x[1],
                      self.workspace_bounds_y[1], self.workspace_bounds_z[1]]
        bounds = np.array([min_coords, max_coords], dtype=np.float32)
        color_rgba = np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32)
        try:
            self.scene.draw_debug_box(bounds, color=color_rgba, wireframe=True)
        except Exception as e:
            print(f"Error drawing workspace debug box: {e}")

    def _generate_target_in_shelf_batched(self) -> np.ndarray:
        self.target_position_world_batch.fill(0.0)
        for i in range(self.num_envs):
            params = self.current_shelf_instance_params_per_env[i]
            open_w, open_d, open_h = params["internal_width"], params[
                "internal_depth"], params["actual_opening_height"]
            target_in_shelf_opening_frame = np.array([random.uniform(-open_w/2*0.7, open_w/2*0.7), random.uniform(
                -open_d/2*0.7, open_d/2*0.3), random.uniform(-open_h/2*0.8, open_h/2*0.8)], dtype=np.float32)
            shelf_assembly_origin_world = params["shelf_assembly_origin_world"]
            shelf_assembly_yaw = params["shelf_assembly_yaw"]
            offset_to_opening_center_local = np.array(
                [0, 0, self.DEFAULT_PLATE_THICKNESS+open_h/2.0], dtype=np.float32)
            shelf_assembly_quat_world_wxyz = np.array([np.cos(
                shelf_assembly_yaw/2.0), 0, 0, np.sin(shelf_assembly_yaw/2.0)], dtype=np.float32)
            shelf_assembly_quat_world_xyzw = shelf_assembly_quat_world_wxyz[[
                1, 2, 3, 0]]
            R_shelf_yaw_world = ScipyRotation.from_quat(
                shelf_assembly_quat_world_xyzw).as_matrix()
            opening_center_world = shelf_assembly_origin_world + \
                (R_shelf_yaw_world @ offset_to_opening_center_local)
            target_offset_world_rotated = R_shelf_yaw_world @ target_in_shelf_opening_frame
            self.target_position_world_batch[i] = opening_center_world + \
                target_offset_world_rotated

        if self.scene and hasattr(self.scene, 'clear_debug_objects'):
            self.scene.clear_debug_objects()

        if self.scene and (self.render_mode == "human" or self._is_recording_active):
            if self._num_envs == 1:
                if self.target_position_world_batch[0] is not None:
                    points_to_draw = [
                        self.target_position_world_batch[0].tolist()]
                    self.scene.draw_debug_spheres(
                        points_to_draw, radius=0.03, color=(0.0, 1.0, 0.0, 0.8))
            elif self._num_envs > 1 and self.target_sphere_entity_type is not None:
                valid_targets = np.asarray(
                    self.target_position_world_batch, dtype=np.float32)
                if valid_targets.ndim == 2 and valid_targets.shape[0] == self._num_envs and valid_targets.shape[1] == 3:
                    target_pos_tensor_all_envs = torch.tensor(
                        valid_targets, device=gs.device, dtype=torch.float32)
                    self.target_sphere_entity_type.set_pos(
                        target_pos_tensor_all_envs)

        if self.render_mode == "human" and self.scene and self._num_envs == 1:
            self._draw_workspace_bounds()
        return self.target_position_world_batch

    def _get_obs_batched(self) -> VecEnvObs:
        arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz = self._get_robot_state_parts_batched()
        robot_state_flat_batch = np.concatenate(
            [arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz], axis=1).astype(np.float32)
        relative_target_pos_batch = (
            self.target_position_world_batch - ee_pos).astype(np.float32)
        flat_shelf_component_params_batch = self.shelf_component_params_batch.reshape(
            self.num_envs, -1)
        return {
            "robot_state_flat": robot_state_flat_batch,
            "relative_target_pos": relative_target_pos_batch,
            "shelf_component_params": flat_shelf_component_params_batch.astype(np.float32)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> VecEnvObs:
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
            raise RuntimeError("Genesis scene is not initialized.")
        self.current_step_per_env.fill(0)
        initial_qpos_batch = np.tile(
            self.FRANKA_DEFAULT_INITIAL_QPOS, (self.num_envs, 1))
        initial_qvel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_TOTAL_JOINTS), dtype=np.float32)
        qpos_tensor = torch.tensor(
            initial_qpos_batch, device=gs.device, dtype=torch.float32)
        qvel_tensor = torch.tensor(
            initial_qvel_batch, device=gs.device, dtype=torch.float32)
        if self.franka_entity:
            self.franka_entity.set_dofs_position(
                qpos_tensor, self.franka_all_dof_indices_local)
            self.franka_entity.set_dofs_velocity(
                qvel_tensor, self.franka_all_dof_indices_local)
        self.prev_joint_vel_batch.fill(0.0)
        # MOD: Call this before generating target if shelf is included
        self._build_shelf_structure_and_populate_params_batched()
        self._generate_target_in_shelf_batched()
        obs_batch = self._get_obs_batched()
        self.buf_infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.buf_infos[i]["shelf_config"] = self.current_shelf_config_key_per_env[i]
            self.buf_infos[i]["include_shelf"] = self.include_shelf
            self.buf_infos[i]["randomize_shelf"] = self.randomize_shelf_config
            # Log initial diagnostic values
            if obs_batch["relative_target_pos"] is not None and len(obs_batch["relative_target_pos"]) > i:
                # MOD: Using slash for W&B grouping
                self.buf_infos[i]["diagnostics/distance_to_target"] = float(
                    np.linalg.norm(obs_batch["relative_target_pos"][i]))
                self.buf_infos[i]["diagnostics/ee_pos_x"] = float(
                    obs_batch["robot_state_flat"][i, self.FRANKA_NUM_ARM_JOINTS*2])
                self.buf_infos[i]["diagnostics/ee_pos_y"] = float(
                    obs_batch["robot_state_flat"][i, self.FRANKA_NUM_ARM_JOINTS*2+1])
                self.buf_infos[i]["diagnostics/ee_pos_z"] = float(
                    obs_batch["robot_state_flat"][i, self.FRANKA_NUM_ARM_JOINTS*2+2])
                self.buf_infos[i]["diagnostics/target_pos_x"] = float(
                    self.target_position_world_batch[i, 0])
                self.buf_infos[i]["diagnostics/target_pos_y"] = float(
                    self.target_position_world_batch[i, 1])
                self.buf_infos[i]["diagnostics/target_pos_z"] = float(
                    self.target_position_world_batch[i, 2])
        return obs_batch

    def step_async(
        self, actions: np.ndarray) -> None: self.actions_buffer = actions.copy()

    def _calculate_rewards_and_dones(self, actions_clipped_batch: np.ndarray, arm_joint_pos_batch: np.ndarray, arm_joint_vel_batch: np.ndarray, ee_pos_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        dist_to_target_batch = np.linalg.norm(
            ee_pos_batch - self.target_position_world_batch, axis=1)

        reward_dist = -self.k_dist_reward * dist_to_target_batch
        penalty_time = -self.k_time_penalty * \
            np.ones(self.num_envs, dtype=np.float32)
        penalty_action_mag = -self.k_action_penalty * \
            np.sum(np.square(actions_clipped_batch), axis=1)

        penalty_joint_limit = np.zeros(self.num_envs, dtype=np.float32)
        near_limit_thresh = 0.05
        for i in range(self.FRANKA_NUM_ARM_JOINTS):
            penalty_joint_limit -= self.k_joint_limit_penalty * \
                np.maximum(
                    0, (self.FRANKA_QPOS_LOWER[i] + near_limit_thresh) - arm_joint_pos_batch[:, i])
            penalty_joint_limit -= self.k_joint_limit_penalty * \
                np.maximum(
                    0, arm_joint_pos_batch[:, i] - (self.FRANKA_QPOS_UPPER[i] - near_limit_thresh))

        penalty_collision = np.zeros(self.num_envs, dtype=np.float32)
        terminated_collision = np.zeros(self.num_envs, dtype=bool)
        if self.scene and self.scene.is_built and self.franka_entity:
            contacts = self.franka_entity.get_contacts()
            if contacts and 'valid_mask' in contacts:
                valid_mask = to_numpy(contacts['valid_mask'])
                for i_env in range(self.num_envs):
                    if np.any(valid_mask[i_env, :]):
                        penalty_collision[i_env] = -self.k_collision_penalty
                        terminated_collision[i_env] = True

        reward_success = np.zeros(self.num_envs, dtype=np.float32)
        terminated_success = dist_to_target_batch < self.success_threshold
        reward_success[terminated_success] = self.success_reward

        penalty_accel = np.zeros(self.num_envs, dtype=np.float32)
        if self.k_accel_penalty > 0 and self.dt > 0:
            acceleration_sq_sum = np.sum(np.square((arm_joint_vel_batch -
                                                    self.prev_joint_vel_batch) / self.dt), axis=1)
            penalty_accel = -self.k_accel_penalty * acceleration_sq_sum

        rewards = reward_dist + penalty_time + penalty_action_mag + \
            penalty_joint_limit + penalty_collision + reward_success + penalty_accel

        terminated = np.logical_or(terminated_collision, terminated_success)
        truncated = self.current_step_per_env >= self.max_steps
        dones = np.logical_or(terminated, truncated)

        infos = []
        for i in range(self.num_envs):
            info = {
                "is_success": terminated_success[i],
                # MOD: Using slash for W&B grouping
                "diagnostics/distance_to_target": float(dist_to_target_batch[i]),
                "diagnostics/collision_detected_term": terminated_collision[i],
                "current_step": self.current_step_per_env[i],
                "shelf_config": self.current_shelf_config_key_per_env[i],
                "include_shelf": self.include_shelf,
                "randomize_shelf": self.randomize_shelf_config,
                "rewards/distance": float(reward_dist[i]),
                "rewards/time_penalty": float(penalty_time[i]),
                "rewards/action_penalty": float(penalty_action_mag[i]),
                "rewards/joint_limit_penalty": float(penalty_joint_limit[i]),
                "rewards/collision_penalty": float(penalty_collision[i]),
                "rewards/success": float(reward_success[i]),
                "rewards/accel_penalty": float(penalty_accel[i]),
                "diagnostics/ee_pos_x": float(ee_pos_batch[i, 0]),
                "diagnostics/ee_pos_y": float(ee_pos_batch[i, 1]),
                "diagnostics/ee_pos_z": float(ee_pos_batch[i, 2]),
                "diagnostics/target_pos_x": float(self.target_position_world_batch[i, 0]),
                "diagnostics/target_pos_y": float(self.target_position_world_batch[i, 1]),
                "diagnostics/target_pos_z": float(self.target_position_world_batch[i, 2]),
                "diagnostics/mean_qpos_j1-3": float(np.mean(arm_joint_pos_batch[i, :3])),
                "diagnostics/mean_qvel_j1-3": float(np.mean(arm_joint_vel_batch[i, :3])),
            }
            if dones[i] and truncated[i] and not terminated[i]:
                info["TimeLimit.truncated"] = True
            infos.append(info)

        return rewards, dones, terminated, infos

    def step_wait(self) -> VecEnvStepReturn:
        if not self.scene or not self.franka_entity:
            raise RuntimeError("Env not initialized.")
        actions_batch = self.actions_buffer
        self.current_step_per_env += 1
        actions_clipped = np.clip(actions_batch, -1.0, 1.0).astype(np.float32)
        scaled_targets = actions_clipped * \
            self.max_allowable_joint_velocity_scale * self.FRANKA_VEL_LIMIT

        actions_tensor = torch.tensor(
            scaled_targets, device=gs.device, dtype=torch.float32)
        self.franka_entity.control_dofs_velocity(
            actions_tensor, self.franka_arm_dof_indices_local)
        self.scene.step()
        if self._is_recording_active and self.video_capture_camera:
            self.video_capture_camera.render()
        arm_qpos, arm_qvel, ee_pos, _ = self._get_robot_state_parts_batched()
        rewards, dones, term, self.buf_infos = self._calculate_rewards_and_dones(
            actions_clipped, arm_qpos, arm_qvel, ee_pos)
        obs = self._get_obs_batched()
        self.prev_joint_vel_batch = arm_qvel.copy()
        for i in range(self.num_envs):
            if dones[i]:
                self.buf_infos[i]["terminal_observation"] = {
                    k: v[i] for k, v in obs.items()}
        return obs, rewards, dones, self.buf_infos

    def render(self, mode: Optional[str] = 'human') -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        if mode == "human" and self.render_mode == "human":
            if self.scene:
                if self._num_envs == 1:
                    if hasattr(self.scene, "draw_debug_spheres") and self.target_position_world_batch[0] is not None:
                        points_to_draw = [
                            self.target_position_world_batch[0].tolist()]
                        self.scene.draw_debug_spheres(
                            points_to_draw, radius=0.03, color=(0.0, 1.0, 0.0, 0.8))
                elif self._num_envs > 1 and self.target_sphere_entity_type is not None:
                    valid_targets = np.asarray(
                        self.target_position_world_batch, dtype=np.float32)
                    if valid_targets.ndim == 2 and valid_targets.shape[0] == self._num_envs and valid_targets.shape[1] == 3:
                        target_pos_tensor_all_envs = torch.tensor(
                            valid_targets, device=gs.device, dtype=torch.float32)
                        self.target_sphere_entity_type.set_pos(
                            target_pos_tensor_all_envs)
                self._draw_workspace_bounds()
            return None
        elif mode == "rgb_array":
            try:
                if self._num_envs > 1 and self.target_sphere_entity_type is not None:
                    valid_targets = np.asarray(
                        self.target_position_world_batch, dtype=np.float32)
                    if valid_targets.ndim == 2 and valid_targets.shape[0] == self._num_envs and valid_targets.shape[1] == 3:
                        target_pos_tensor_all_envs = torch.tensor(
                            valid_targets, device=gs.device, dtype=torch.float32)
                        self.target_sphere_entity_type.set_pos(
                            target_pos_tensor_all_envs)

                rgb_output, _, _, _ = self.video_capture_camera.render(
                    rgb=True, depth=False, segmentation=False, normal=False)
                return rgb_output[0] if isinstance(rgb_output, list) and rgb_output else rgb_output
            except Exception:
                return None
        return None

    def close(self) -> None:
        if self.video_capture_camera:
            try:
                if hasattr(self.video_capture_camera, 'close') and callable(self.video_capture_camera.close):
                    self.video_capture_camera.close()
            except Exception:
                pass
            finally:
                self.video_capture_camera = None
        if self.scene:
            try:
                if hasattr(self.scene, 'close') and callable(self.scene.close):
                    self.scene.close()
                elif hasattr(self.scene, 'shutdown') and callable(self.scene.shutdown):
                    self.scene.shutdown()
            except Exception:
                pass
            finally:
                self.scene = None
        self.target_sphere_entity_type = None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        data = getattr(self, attr_name)
        return [data for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any,
                 indices: VecEnvIndices = None) -> None: setattr(self, attr_name, value)

    def env_method(self, method_name: str, *args, indices: VecEnvIndices = None, **kwargs) -> List[Any]:
        method = getattr(self, method_name)
        return [method(*args, **kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [isinstance(self, wrapper_class) for _ in self._get_indices(indices)]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed for _ in range(self.num_envs)]

    def start_video_recording(self, env_idx_to_focus: int = 0) -> bool:
        if not self.video_capture_camera:
            self._is_recording_active = False
            return False
        try:
            self.video_capture_camera.start_recording()
            self._is_recording_active = True
            if self.scene and hasattr(self.scene, 'clear_debug_objects'):
                self.scene.clear_debug_objects()

            if self.scene:
                if self._num_envs == 1:
                    if 0 <= env_idx_to_focus < self._num_envs and self.target_position_world_batch[env_idx_to_focus] is not None:
                        points_to_draw = [
                            self.target_position_world_batch[env_idx_to_focus].tolist()]
                        self.scene.draw_debug_spheres(
                            points_to_draw, radius=0.03, color=(0.0, 1.0, 0.0, 0.8))
                elif self._num_envs > 1 and self.target_sphere_entity_type is not None:
                    valid_targets = np.asarray(
                        self.target_position_world_batch, dtype=np.float32)
                    if valid_targets.ndim == 2 and valid_targets.shape[0] == self._num_envs and valid_targets.shape[1] == 3:
                        target_pos_tensor_all_envs = torch.tensor(
                            valid_targets, device=gs.device, dtype=torch.float32)
                        self.target_sphere_entity_type.set_pos(
                            target_pos_tensor_all_envs)

                self._draw_workspace_bounds()
            return True
        except Exception as e:
            print(f"Error in start_video_recording: {e}")
            self._is_recording_active = False
            return False

    def stop_video_recording(self, save_dir: str, filename: str, fps: int = 30) -> Optional[str]:
        if not self._is_recording_active or not self.video_capture_camera:
            self._is_recording_active = False
            return None
        try:
            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(save_dir, filename)
            self.video_capture_camera.stop_recording(
                save_to_filename=full_path, fps=fps)
            return full_path
        except Exception as e:
            print(f"Error in stop_video_recording: {e}")
            return None
        finally:
            self._is_recording_active = False


# --- Main block for testing the environment ---
if __name__ == '__main__':
    print("--- FrankaShelfEnv Test Script (Shelf Structure with Parameter Observation) ---")
    gs_initialized_in_main = False
    try:
        if not (hasattr(gs, '_is_initialized') and gs._is_initialized):
            gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)
            gs_initialized_in_main = True
        print("Genesis initialized successfully for testing.")
    except Exception as e_init:
        print(
            f"CRITICAL ERROR: Failed to initialize Genesis for testing: {e_init}\nExiting.")
        sys.exit(1)

    num_test_envs_for_run = 5  # MOD: Changed to 1 for more focused testing of info dict
    print(f"\n\n--- TESTING WITH NUM_ENVS = {num_test_envs_for_run} ---")
    NUM_TEST_ENVS = num_test_envs_for_run
    env = None
    try:
        test_scenario = 3  # Test with no shelf for simplicity of reward debugging

        if test_scenario == 1:
            print(
                f"\nCreating FrankaShelfEnv (Shelf ON, Randomized) num_envs={NUM_TEST_ENVS}, render_mode='human'")
            env = FrankaShelfEnv(num_envs=NUM_TEST_ENVS, render_mode="human",
                                 include_shelf=True, randomize_shelf_config=True)
        elif test_scenario == 2:
            print(
                f"\nCreating FrankaShelfEnv (Shelf ON, NOT Randomized) num_envs={NUM_TEST_ENVS}, render_mode='human'")
            env = FrankaShelfEnv(num_envs=NUM_TEST_ENVS, render_mode="human",
                                 include_shelf=True, randomize_shelf_config=False)
        elif test_scenario == 3:
            print(
                f"\nCreating FrankaShelfEnv (Shelf OFF) num_envs={NUM_TEST_ENVS}, render_mode='human'")
            env = FrankaShelfEnv(num_envs=NUM_TEST_ENVS, render_mode="human",
                                 include_shelf=False, randomize_shelf_config=False)
        else:
            print("Invalid test_scenario.")
            sys.exit(1)

        if env is None:
            print("ERROR: Environment was not created.")
            # sys.exit(1) # Exit if env is None after conditional creation

        print(
            f"FrankaShelfEnv created. Obs Space: {env.observation_space}, Act Space: {env.action_space}")
        print(
            f"  Include Shelf: {env.include_shelf}, Randomize Shelf: {env.randomize_shelf_config}")

        print("\nResetting environment...")
        obs_batch = env.reset(seed=42)
        print("Sample observation (first env if exists):")
        if obs_batch is not None and NUM_TEST_ENVS > 0:
            for key, value in obs_batch.items():  # type: ignore
                print(f"  {key}: shape {value[0].shape if isinstance(value, np.ndarray) and value.ndim > 1 and value.shape[0] > 0 else np.array(value[0]).shape if NUM_TEST_ENVS==1 else 'N/A for multi-env sample'}, sample data: {value[0][:5] if isinstance(value[0], np.ndarray) and value[0].ndim >0 and value[0].size > 5 else value[0] if NUM_TEST_ENVS==1 else 'N/A for multi-env sample'}")
        print(
            f"  Initial shelf config key: {env.current_shelf_config_key_per_env[0] if NUM_TEST_ENVS > 0 else 'N/A'}")
        if NUM_TEST_ENVS > 0 and env.buf_infos[0]:
            print(f"  Initial info for env 0: {env.buf_infos[0]}")

        if env.scene and env.render_mode == "human":
            print("Stepping scene for initial view (simulating render loop)...")
            for _ in range(30):
                if env.scene:
                    env.scene.step()

        print("\nRunning a short loop with random actions for visualization...")
        for step_num in range(300):  # MOD: Shorter loop for testing info dict
            actions = np.array([env.action_space.sample()
                                for _ in range(env.num_envs)])
            env.step_async(actions)
            next_obs_batch, rewards_b, dones_b, infos_b = env.step_wait()

            if env.render_mode == "human":
                env.render(mode="human")

            if (step_num + 1) % 5 == 0:  # MOD: Log more frequently
                print(
                    f"  Step {step_num+1} completed. Reward: {rewards_b[0]:.2f}, Done: {dones_b[0]}, Success: {infos_b[0].get('is_success', False) if NUM_TEST_ENVS > 0 else 'N/A'}")
                if NUM_TEST_ENVS > 0 and infos_b[0]:
                    # Print the full info dict for debugging
                    print(f"    Info env 0: {infos_b[0]}")
                if dones_b[0] and NUM_TEST_ENVS > 0:
                    print(f"  Episode ended for env 0. Info: {infos_b[0]}")
        print("\nBasic visualization loop finished.")
    except Exception as e_runtime:
        print(
            f"ERROR during FrankaShelfEnv example usage (NUM_ENVS={NUM_TEST_ENVS}): {e_runtime}")
        traceback.print_exc()
    finally:
        if env:
            print("Closing environment...")
            env.close()

    if gs_initialized_in_main and hasattr(gs, 'shutdown') and callable(gs.shutdown):
        print("Shutting down Genesis post-testing.")
        gs.shutdown()
    print("--- FrankaShelfEnv Test Script Finished ---")
