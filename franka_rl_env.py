"""
Defines the FrankaShelfEnv, a vectorized reinforcement learning environment
for a Franka Emika Panda robot, adapted for the RSL-RL library.

The environment involves tasks like reaching specific poses. The observation
space is a flattened tensor including robot state and target position.
The action space can be configured for either velocity or torque control.
It uses the Genesis simulation engine for physics and rendering.
"""
import os
import sys
import random
import math
from typing import List, Any, Optional, Union, Dict, Tuple
import traceback
from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

import genesis as gs  # type: ignore
from rsl_rl.env import VecEnv

from curriculum import LinearCurriculum, CurriculumConfig


def to_numpy(data: Union[torch.Tensor, np.ndarray, List, Tuple]) -> np.ndarray:
    """
    Converts input data (PyTorch Tensor, list, tuple) to a NumPy array.
    If the input is a PyTorch Tensor on a CUDA device, it's moved to the CPU.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy() if data.is_cuda else data.numpy()
    return np.asarray(data, dtype=np.float32)


class FrankaShelfEnv(VecEnv):
    """
    A vectorized environment for a Franka Emika Panda robot, adapted for RSL-RL.

    Observations are fixed and include:
    - Base robot state (joint positions, velocities, EE pose).
    - Relative target position.
    - End-effector linear and angular velocities.
    - History of previous actions.

    Actions are continuous joint velocity or torque commands.

    Rewards are shaped to encourage reaching a target smoothly and stably,
    with configurable penalties for high acceleration and jerk. Joint and EE
    velocity penalties are dynamically scaled based on proximity to the target.
    """
    # https://frankaemika.github.io/docs/control_parameters.html
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
    FRANKA_TORQUE_LIMIT: np.ndarray = np.array(
        [87, 87, 87, 87, 12, 12, 12], dtype=np.float32)
    ROBOT_EE_LINK_NAME: str = "hand"

    FRANKA_QPOS_RESET_NOISE_RANGES: np.ndarray = np.array([
        (-1.0, 1.0), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2),
        (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)
    ], dtype=np.float32)

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
                 num_envs: int = 1,
                 max_steps_per_episode: int = 1000,
                 env_spacing: Tuple[float, float] = (1.5, 1.5),
                 workspace_bounds_xyz: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
                     (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.5)),
                 dt: float = 0.01,
                 control_mode: str = 'velocity',
                 num_actions_history: int = 1,
                 # --- Reward & Curriculum Config ---
                 franka_xml_path: str = 'xml/franka_emika_panda/panda.xml',
                 k_dist_reward: float = 1.0,
                 k_joint_limit_penalty: float = 10.0,
                 k_collision_penalty: float = 100.0,
                 success_reward_val: float = 200.0,
                 proximity_vel_penalty_max_scale: float = 4.0,
                 proximity_vel_penalty_dist_threshold: float = 0.2,
                 threshold_curriculum_cfg: dict = None,
                 joint_velocity_penalty_curriculum_cfg: dict = None,
                 ee_velocity_penalty_curriculum_cfg: dict = None,
                 action_penalty_curriculum_cfg: dict = None,
                 accel_penalty_curriculum_cfg: dict = None,
                 jerk_penalty_curriculum_cfg: dict = None,
                 upright_bonus_curriculum_cfg: dict = None,
                 min_episode_length_for_success_metric: int = 10,
                 # --- Rendering & Video ---
                 video_camera_pos: Tuple[float,
                                         float, float] = (1.8, -1.8, 2.0),
                 video_camera_lookat: Tuple[float,
                                            float, float] = (0.3, 0.0, 0.5),
                 video_camera_fov: float = 45, video_res: Tuple[int, int] = (960, 640),
                 include_shelf: bool = True,
                 randomize_shelf_config: bool = True,
                 device: str = 'cpu',
                 seed: int = 42,
                 render: bool = False
                 ):
        self.device = device
        self.np_random = None
        self.seed(seed)

        self.render = render
        self.num_envs = num_envs
        self.include_shelf = include_shelf
        self.randomize_shelf_config = randomize_shelf_config

        # --- Control Mode Selection ---
        self.control_mode = control_mode
        if self.control_mode not in ['velocity', 'torque']:
            raise ValueError(
                f"Invalid control_mode: {self.control_mode}. Must be 'velocity' or 'torque'.")

        # --- RSL-RL specific attributes ---
        self.num_actions_history = num_actions_history
        robot_state_flat_dim = self.FRANKA_NUM_ARM_JOINTS * \
            2 + 3 + 4  # qpos, qvel, ee_pos, ee_quat
        relative_target_pos_dim = 3
        ee_vel_dim = 6
        action_history_dim = self.FRANKA_NUM_ARM_JOINTS * \
            self.num_actions_history if self.num_actions_history > 0 else 0

        self.num_obs = robot_state_flat_dim + \
            relative_target_pos_dim + ee_vel_dim + action_history_dim
        self.num_privileged_obs = 0
        self.num_actions = self.FRANKA_NUM_ARM_JOINTS
        self.max_episode_length = max_steps_per_episode

        self.env_spacing = env_spacing
        self.workspace_bounds_x, self.workspace_bounds_y, self.workspace_bounds_z = workspace_bounds_xyz
        self.dt = dt
        self.franka_xml_path = franka_xml_path
        self.k_dist_reward = k_dist_reward
        self.k_joint_limit_penalty = k_joint_limit_penalty
        self.k_collision_penalty = k_collision_penalty
        self.success_reward = success_reward_val
        self.proximity_vel_penalty_max_scale = proximity_vel_penalty_max_scale
        self.proximity_vel_penalty_dist_threshold = proximity_vel_penalty_dist_threshold

        # --- Initialize Curriculum Components ---
        required_curricula = [
            threshold_curriculum_cfg, joint_velocity_penalty_curriculum_cfg,
            ee_velocity_penalty_curriculum_cfg, action_penalty_curriculum_cfg,
            accel_penalty_curriculum_cfg, jerk_penalty_curriculum_cfg,
            upright_bonus_curriculum_cfg
        ]
        if any(cfg is None for cfg in required_curricula):
            raise ValueError("All curriculum configurations must be provided.")

        self.threshold_curriculum = LinearCurriculum(
            CurriculumConfig(**threshold_curriculum_cfg))
        self.joint_velocity_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**joint_velocity_penalty_curriculum_cfg))
        self.ee_velocity_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**ee_velocity_penalty_curriculum_cfg))
        self.action_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**action_penalty_curriculum_cfg))
        self.accel_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**accel_penalty_curriculum_cfg))
        self.jerk_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**jerk_penalty_curriculum_cfg))
        self.upright_bonus_curriculum = LinearCurriculum(
            CurriculumConfig(**upright_bonus_curriculum_cfg))

        self.min_episode_length_for_success_metric = min_episode_length_for_success_metric
        self.success_buffer = deque(maxlen=100 * self.num_envs)
        self.current_success_rate = 0.0

        self._define_shelf_configurations()
        self.current_shelf_config_key_per_env: List[str] = [""] * self.num_envs
        self.current_shelf_instance_params_per_env: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)]

        # --- Initialize Genesis Scene ---
        sim_viewer_options = gs.options.ViewerOptions(camera_pos=video_camera_pos, camera_lookat=video_camera_lookat, camera_fov=video_camera_fov,
                                                      res=video_res, max_FPS=60)
        sim_options = gs.options.SimOptions(dt=self.dt)
        self.scene = gs.Scene(viewer_options=sim_viewer_options,
                              sim_options=sim_options, show_viewer=self.render)

        # Add a dedicated, non-GUI camera for video recording
        self.video_camera_params = {'res': video_res, 'pos': video_camera_pos,
                                    'lookat': video_camera_lookat, 'fov': video_camera_fov}
        self.video_capture_camera = self.scene.add_camera(
            res=self.video_camera_params['res'], pos=self.video_camera_params['pos'], lookat=self.video_camera_params['lookat'], fov=self.video_camera_params['fov'], GUI=False)
        self._is_recording = False

        self.plane_entity = self.scene.add_entity(gs.morphs.Plane())
        self.franka_entity = self.scene.add_entity(
            gs.morphs.MJCF(file=self.franka_xml_path))

        # Get the index of the end-effector link for efficient velocity queries
        self.ee_link_idx = self.franka_entity.get_link(
            self.ROBOT_EE_LINK_NAME).idx

        self.shelf_component_entities: List[Any] = []
        if self.include_shelf:
            self.shelf_component_entities = [self.scene.add_entity(gs.morphs.Box(pos=(0, -10 - i * 0.5, 0), quat=(1, 0, 0, 0), size=tuple(
                s), fixed=True, collision=True, visualization=True)) for i, s in enumerate(self.FIXED_COMPONENT_SIZES)]
        else:
            self.shelf_component_entities = []

        self.target_sphere_entity_type: Optional[Any] = None
        if self.scene and self.num_envs > 1:
            self.target_sphere_entity_type = self.scene.add_entity(gs.morphs.Sphere(
                pos=(0, -20, 0), radius=0.03, visualization=True, collision=False, fixed=True))

        if not self.scene.is_built:
            self.scene.build(n_envs=self.num_envs,
                             env_spacing=self.env_spacing)

        # --- Buffers and State Variables ---
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        # For logging episode stats
        self.episode_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.ep_infos = []

        # General purpose dictionary for step-wise info
        self.extras = {}

        self.shelf_component_params_batch = np.zeros(
            (self.num_envs, self.SHELF_NUM_COMPONENTS, self.NUM_PARAMS_PER_SHELF_COMPONENT), dtype=np.float32)
        self.target_position_world_batch = np.zeros(
            (self.num_envs, 3), dtype=np.float32)

        # Buffers for new reward terms and observations
        self.prev_joint_vel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)
        self.prev_joint_accel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)
        self.prev_actions_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS * self.num_actions_history), dtype=np.float32)

        self.franka_all_dof_indices_local = np.array([self.franka_entity.get_joint(
            name).dof_idx_local for name in self.FRANKA_JOINT_NAMES], dtype=np.int32)
        self.franka_arm_dof_indices_local = self.franka_all_dof_indices_local[
            :self.FRANKA_NUM_ARM_JOINTS]

        # Initialize environment state
        self.reset()

    def start_video_recording(self):
        """Enable video recording."""
        if self.video_capture_camera:
            self._is_recording = True
            self.video_capture_camera.start_recording()
            print("Video recording started.")

    def stop_video_recording(self, file_path: str):
        """
        Stop video recording and save the video to the specified path.
        """
        if self.video_capture_camera and self._is_recording:
            try:
                self.video_capture_camera.stop_recording(
                    save_to_filename=file_path)
                print(f"Video saved to {file_path}")
            except Exception as e:
                print(f"Error stopping video recording: {e}")
            finally:
                self._is_recording = False

    def get_observations(self):
        """
        Returns the observation buffer and the extras dictionary.
        This is the format expected by the RSL-RL OnPolicyRunner.
        """
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def get_episode_infos(self):
        """
        Returns the episode information list and clears the internal buffer.
        """
        infos_to_return = self.ep_infos.copy()
        self.ep_infos.clear()
        return infos_to_return

    def _define_shelf_configurations(self) -> None:
        self.shelf_configurations: Dict[str, Dict[str, Any]] = {
            "default_center_reach": {"name": "default_center_reach", "base_pos_range_x": [(-0.6, -0.35), (0.35, 0.6)], "base_pos_range_y": (0.0, 0.0), "base_pos_range_z": (0.3, 0.3)},
            "high_center_reach": {"name": "high_center_reach", "base_pos_range_x": [(-0.6, -0.35), (0.35, 0.6)], "base_pos_range_y": (-0.4, 0.4), "base_pos_range_z": (0.4, 0.65)},
            "low_forward_reach": {"name": "low_forward_reach", "base_pos_range_x": [(-0.60, -0.35), (0.35, 0.60)], "base_pos_range_y": (-0.4, 0.4), "base_pos_range_z": (0.1, 0.30)},
            "mid_side_reach_right": {"name": "mid_side_reach_right", "base_pos_range_x": [(0.3, 0.5)], "base_pos_range_y": (-0.6, 0.6), "base_pos_range_z": (0.2, 0.5)},
            "mid_side_reach_left": {"name": "mid_side_reach_left", "base_pos_range_x": [(-0.5, -0.3)], "base_pos_range_y": (-0.6, -0.6), "base_pos_range_z": (0.2, 0.5)}
        }
        self.shelf_config_keys_list: List[str] = list(
            self.shelf_configurations.keys())
        self.default_shelf_config_key: str = "default_center_reach"
        buffer_half_width = 0.2
        for key, config_details in self.shelf_configurations.items():
            for x_range_tuple in config_details["base_pos_range_x"]:
                if not (x_range_tuple[1] <= -buffer_half_width or x_range_tuple[0] >= buffer_half_width):
                    raise ValueError(
                        f"Configuration '{key}' x_range {x_range_tuple} violates the buffer zone (-{buffer_half_width}, {buffer_half_width}).")

    def _get_robot_state_parts_batched(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves batched kinematic data for the robot.

        Returns:
            A tuple containing:
            - arm_joint_pos_batch (np.ndarray): Joint positions.
            - arm_joint_vel_batch (np.ndarray): Joint velocities.
            - ee_pos_batch (np.ndarray): End-effector position.
            - ee_orient_quat_wxyz_batch (np.ndarray): End-effector orientation.
            - ee_vel_batch (np.ndarray): End-effector linear and angular velocity.
        """
        joint_pos_all_batch = to_numpy(
            self.franka_entity.get_dofs_position(self.franka_all_dof_indices_local))
        joint_vel_all_batch = to_numpy(
            self.franka_entity.get_dofs_velocity(dofs_idx_local=self.franka_all_dof_indices_local))
        arm_joint_pos_batch = joint_pos_all_batch[:,
                                                  :self.FRANKA_NUM_ARM_JOINTS]
        arm_joint_vel_batch = joint_vel_all_batch[:,
                                                  :self.FRANKA_NUM_ARM_JOINTS]

        ee_link = self.franka_entity.get_link(self.ROBOT_EE_LINK_NAME)
        ee_pos_batch = to_numpy(ee_link.get_pos())
        ee_orient_quat_wxyz_batch = to_numpy(ee_link.get_quat())

        # Get all link velocities and select the one for the end-effector
        # Shape: (num_envs, num_links, 3)
        all_links_vel = self.franka_entity.get_links_vel()
        # Shape: (num_envs, num_links, 3)
        all_links_ang = self.franka_entity.get_links_ang()
        ee_linear_vel = all_links_vel[:, self.ee_link_idx, :]
        ee_angular_vel = all_links_ang[:, self.ee_link_idx, :]

        ee_vel_tensor = torch.cat(
            [ee_linear_vel, ee_angular_vel], dim=1)  # Shape: (num_envs, 6)
        ee_vel_batch = to_numpy(ee_vel_tensor)

        return arm_joint_pos_batch, arm_joint_vel_batch, ee_pos_batch, ee_orient_quat_wxyz_batch, ee_vel_batch

    def _build_shelf_structure_and_populate_params_batched(self, env_ids: Union[np.ndarray, List[int]]) -> None:
        if not len(env_ids):
            return

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids

        self.shelf_component_params_batch[env_ids] = 0.0
        shelf_assembly_origins_world_batch = np.zeros(
            (len(env_ids_list), 3), dtype=np.float32)
        shelf_assembly_yaws_batch = np.zeros(
            len(env_ids_list), dtype=np.float32)

        for i, env_idx in enumerate(env_ids_list):
            config_key = random.choice(
                self.shelf_config_keys_list) if self.randomize_shelf_config else self.default_shelf_config_key
            self.current_shelf_config_key_per_env[env_idx] = config_key
            config = self.shelf_configurations[config_key]

            if self.randomize_shelf_config:
                x_ranges_for_config = config["base_pos_range_x"]
                chosen_x_sub_range = random.choice(x_ranges_for_config)
                shelf_assembly_origins_world_batch[i, 0] = random.uniform(
                    *chosen_x_sub_range)
                shelf_assembly_origins_world_batch[i, 1] = random.uniform(
                    *config["base_pos_range_y"])
                shelf_assembly_origins_world_batch[i, 2] = random.uniform(
                    *config["base_pos_range_z"])
            else:
                x_ranges_for_config = config["base_pos_range_x"]
                shelf_assembly_origins_world_batch[i, 0] = max(
                    r[1] for r in x_ranges_for_config)
                shelf_assembly_origins_world_batch[i,
                                                   1] = config["base_pos_range_y"][1]
                shelf_assembly_origins_world_batch[i,
                                                   2] = config["base_pos_range_z"][1]

            if "center" in config["name"] or "forward" in config["name"]:
                shelf_assembly_yaws_batch[i] = math.atan2(
                    -shelf_assembly_origins_world_batch[i, 1], -shelf_assembly_origins_world_batch[i, 0]) - np.pi / 2.0
            elif "right" in config["name"]:
                shelf_assembly_yaws_batch[i] = np.pi
            elif "left" in config["name"]:
                shelf_assembly_yaws_batch[i] = 0.0
            else:
                shelf_assembly_yaws_batch[i] = 0.0

            self.current_shelf_instance_params_per_env[env_idx] = {
                "shelf_assembly_origin_world": shelf_assembly_origins_world_batch[i].copy(),
                "shelf_assembly_yaw": shelf_assembly_yaws_batch[i],
                "actual_opening_height": self.DEFAULT_ACTUAL_OPENING_HEIGHT,
                "internal_width": self.DEFAULT_PLATE_WIDTH - 2 * self.DEFAULT_WALL_THICKNESS,
                "internal_depth": self.DEFAULT_PLATE_DEPTH - self.DEFAULT_WALL_THICKNESS,
            }

        cos_yaw_half_batch = np.cos(shelf_assembly_yaws_batch / 2.0)
        sin_yaw_half_batch = np.sin(shelf_assembly_yaws_batch / 2.0)
        shelf_assembly_world_quats_batch_wxyz = np.zeros(
            (len(env_ids_list), 4), dtype=np.float32)
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

        for comp_idx in range(self.SHELF_NUM_COMPONENTS):
            fixed_size_of_this_component = self.FIXED_COMPONENT_SIZES[comp_idx]
            batched_final_comp_pos_world = np.zeros(
                (len(env_ids_list), 3), dtype=np.float32)

            for i, env_idx in enumerate(env_ids_list):
                assembly_quat_xyzw_i = shelf_assembly_world_quats_batch_wxyz[i, [
                    1, 2, 3, 0]]
                R_shelf_assembly_yaw_i = ScipyRotation.from_quat(
                    assembly_quat_xyzw_i).as_matrix()
                rotated_offset_world_i = R_shelf_assembly_yaw_i @ local_component_offsets[comp_idx]
                batched_final_comp_pos_world[i] = shelf_assembly_origins_world_batch[i] + \
                    rotated_offset_world_i

                self.shelf_component_params_batch[env_idx,
                                                  comp_idx, 0:3] = batched_final_comp_pos_world[i]
                self.shelf_component_params_batch[env_idx, comp_idx,
                                                  3:7] = shelf_assembly_world_quats_batch_wxyz[i]
                self.shelf_component_params_batch[env_idx,
                                                  comp_idx, 7:10] = fixed_size_of_this_component

            if self.include_shelf and comp_idx < len(self.shelf_component_entities):
                component_entity = self.shelf_component_entities[comp_idx]
                all_pos = component_entity.get_pos()
                all_quat = component_entity.get_quat()

                pos_tensor = torch.tensor(
                    batched_final_comp_pos_world, device=self.device, dtype=torch.float32)
                quat_tensor = torch.tensor(
                    shelf_assembly_world_quats_batch_wxyz, device=self.device, dtype=torch.float32)

                all_pos[env_ids] = pos_tensor
                all_quat[env_ids] = quat_tensor

                component_entity.set_pos(all_pos)
                component_entity.set_quat(all_quat)

    def _draw_target_spheres(self):
        """Draws the target spheres in the simulation viewer."""
        # Use the current success threshold from the curriculum to set the radius
        # of the visual target sphere, so we can see the goal shrinking.
        current_radius = self.threshold_curriculum.current_value

        if (self.render or self._is_recording) and self.scene:
            if self.num_envs == 1:
                # If there's only one environment, draw a single debug sphere.
                points_to_draw = [self.target_position_world_batch[0].tolist()]
                self.scene.draw_debug_spheres(
                    points_to_draw, radius=current_radius, color=(0.0, 1.0, 0.0, 0.8))
            elif self.num_envs > 1 and self.target_sphere_entity_type is not None:
                # For multiple environments, update the positions and radius.
                target_pos_tensor = torch.from_numpy(
                    self.target_position_world_batch).to(self.device, dtype=torch.float32)
                self.target_sphere_entity_type.set_pos(target_pos_tensor)

    def _randomize_target_positions(self, env_ids: Union[np.ndarray, List[int]]) -> None:
        """
        Calculates and sets new random target positions for the specified environments.
        This is only called during a reset.
        """
        if not len(env_ids):
            return

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids

        for i, env_idx in enumerate(env_ids_list):
            params = self.current_shelf_instance_params_per_env[env_idx]
            open_w, open_d, open_h = params["internal_width"], params["internal_depth"], params["actual_opening_height"]
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
            self.target_position_world_batch[env_idx] = opening_center_world + \
                target_offset_world_rotated

    def _compute_observations(self) -> None:
        """Computes observations and populates self.obs_buf."""
        arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz, ee_vel = self._get_robot_state_parts_batched()

        # Base robot state
        robot_state_flat_batch = np.concatenate(
            [arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz], axis=1).astype(np.float32)

        # Relative target position
        relative_target_pos_batch = (
            self.target_position_world_batch - ee_pos).astype(np.float32)

        # Assemble the full observation
        obs_parts = [
            robot_state_flat_batch,
            relative_target_pos_batch,
            ee_vel.astype(np.float32)
        ]

        if self.num_actions_history > 0:
            obs_parts.append(self.prev_actions_batch.astype(np.float32))

        obs_np = np.concatenate(obs_parts, axis=-1)
        self.obs_buf = torch.from_numpy(obs_np).to(self.device)
        # Populate the extras dictionary as expected by the runner
        self.extras['observations'] = {}

    def reset(self):
        """Reset all environments."""
        all_env_ids = np.arange(self.num_envs)
        self.reset_idx(all_env_ids)
        self._compute_observations()
        # The runner expects a tuple of (observations, extras)
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: Union[np.ndarray, List[int]]):
        """
        Resets the environment state for the given indices.
        This includes randomizing the target position for the new episode.
        """
        if not len(env_ids):
            return

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids

        # Reset robot state
        initial_qpos = np.tile(
            self.FRANKA_DEFAULT_INITIAL_QPOS, (len(env_ids_list), 1))
        low_noise = self.FRANKA_QPOS_RESET_NOISE_RANGES[:, 0]
        high_noise = self.FRANKA_QPOS_RESET_NOISE_RANGES[:, 1]
        arm_noise = self.np_random.uniform(low_noise, high_noise, size=(
            len(env_ids_list), self.FRANKA_NUM_ARM_JOINTS))
        initial_qpos[:, :self.FRANKA_NUM_ARM_JOINTS] += arm_noise

        for i in range(self.FRANKA_NUM_ARM_JOINTS):
            initial_qpos[:, i] = np.clip(
                initial_qpos[:, i], self.FRANKA_QPOS_LOWER[i] + 0.01, self.FRANKA_QPOS_UPPER[i] - 0.01)

        initial_qvel = np.zeros(
            (len(env_ids_list), self.FRANKA_NUM_TOTAL_JOINTS), dtype=np.float32)

        if self.franka_entity:
            all_qpos = self.franka_entity.get_dofs_position(
                self.franka_all_dof_indices_local)
            all_qvel = self.franka_entity.get_dofs_velocity(
                dofs_idx_local=self.franka_all_dof_indices_local)

            qpos_tensor = torch.tensor(
                initial_qpos, device=self.device, dtype=torch.float32)
            qvel_tensor = torch.tensor(
                initial_qvel, device=self.device, dtype=torch.float32)

            all_qpos[env_ids] = qpos_tensor
            all_qvel[env_ids] = qvel_tensor

            self.franka_entity.set_dofs_position(
                all_qpos, self.franka_all_dof_indices_local)
            self.franka_entity.set_dofs_velocity(
                all_qvel, self.franka_all_dof_indices_local)

        # Reset custom buffers
        self.prev_joint_vel_batch[env_ids] = 0.0
        self.prev_joint_accel_batch[env_ids] = 0.0
        if self.num_actions_history > 0:
            self.prev_actions_batch[env_ids] = 0.0

        # Randomize the shelf configuration for the new episode
        self._build_shelf_structure_and_populate_params_batched(env_ids)

        # Randomize the target position based on the new shelf configuration
        self._randomize_target_positions(env_ids)

        # Draw visualization assets
        self._draw_target_spheres()

        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Compute and return observations
        self._compute_observations()
        return self.obs_buf

    def _calculate_rewards_and_dones(self) -> None:
        """Calculates rewards and dones and populates the buffers."""
        actions_clipped_batch = to_numpy(self.actions)
        arm_joint_pos_batch, arm_joint_vel_batch, ee_pos_batch, _, ee_vel_batch = self._get_robot_state_parts_batched()

        dist_to_target_batch = np.linalg.norm(
            ee_pos_batch - self.target_position_world_batch, axis=1)

        # --- Curriculum Update ---
        # Update success rate based on the buffer.
        if len(self.success_buffer) > 0:
            self.current_success_rate = np.mean(self.success_buffer)

        # Update each curriculum component with the current success rate.
        self.threshold_curriculum.update(self.current_success_rate)
        self.joint_velocity_penalty_curriculum.update(
            self.current_success_rate)
        self.ee_velocity_penalty_curriculum.update(self.current_success_rate)
        self.action_penalty_curriculum.update(self.current_success_rate)
        self.accel_penalty_curriculum.update(self.current_success_rate)
        self.jerk_penalty_curriculum.update(self.current_success_rate)
        self.upright_bonus_curriculum.update(self.current_success_rate)

        # Get the current values from the curriculum objects.
        current_success_threshold = self.threshold_curriculum.current_value
        current_joint_vel_penalty = self.joint_velocity_penalty_curriculum.current_value
        current_ee_vel_penalty = self.ee_velocity_penalty_curriculum.current_value
        current_action_penalty = self.action_penalty_curriculum.current_value
        current_accel_penalty = self.accel_penalty_curriculum.current_value
        current_jerk_penalty = self.jerk_penalty_curriculum.current_value
        current_upright_bonus = self.upright_bonus_curriculum.current_value

        # --- Dones ---
        # The success condition now uses the dynamic threshold from the curriculum.
        terminated_success = dist_to_target_batch < current_success_threshold

        terminated_collision = np.zeros(self.num_envs, dtype=bool)
        if self.scene and self.scene.is_built and self.franka_entity:
            contacts = self.franka_entity.get_contacts()
            if contacts and 'valid_mask' in contacts:
                valid_mask = to_numpy(contacts['valid_mask'])
                terminated_collision = np.any(valid_mask, axis=1)

        terminated = np.logical_or(terminated_collision, terminated_success)
        truncated = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        dones_np = np.logical_or(terminated, truncated)

        # Update success buffer for the next iteration's curriculum calculation.
        done_indices = np.where(dones_np)[0]
        for idx in done_indices:
            if self.episode_length_buf[idx] >= self.min_episode_length_for_success_metric:
                self.success_buffer.append(terminated_success[idx])

        # --- Reward Components ---
        reward_dist = -self.k_dist_reward * dist_to_target_batch

        # Use the current penalty value from the curriculum
        penalty_action_mag = -current_action_penalty * \
            np.sum(np.square(actions_clipped_batch), axis=1)

        out_of_bounds = (arm_joint_pos_batch < self.FRANKA_QPOS_LOWER) | (
            arm_joint_pos_batch > self.FRANKA_QPOS_UPPER)
        penalty_joint_limit = -self.k_joint_limit_penalty * \
            np.sum(out_of_bounds, axis=1)

        penalty_collision = -self.k_collision_penalty * \
            terminated_collision.astype(float)

        # Dynamic velocity penalties based on proximity
        proximity_scaling_factor = np.ones(self.num_envs, dtype=np.float32)
        if self.proximity_vel_penalty_dist_threshold > 0:
            close_mask = dist_to_target_batch < self.proximity_vel_penalty_dist_threshold
            # Linearly scale from 1.0 (at threshold) up to `max_scale` (at dist 0)
            normalized_dist_inv = 1.0 - \
                (dist_to_target_batch[close_mask] /
                 self.proximity_vel_penalty_dist_threshold)
            proximity_scaling_factor[close_mask] += (
                self.proximity_vel_penalty_max_scale - 1.0) * normalized_dist_inv

        joint_vel_magnitude = np.linalg.norm(arm_joint_vel_batch, axis=1)
        penalty_joint_velocity = -current_joint_vel_penalty * \
            joint_vel_magnitude * proximity_scaling_factor

        ee_vel_magnitude = np.linalg.norm(ee_vel_batch, axis=1)
        penalty_ee_velocity = -current_ee_vel_penalty * \
            ee_vel_magnitude * proximity_scaling_factor

        reward_success = self.success_reward * terminated_success.astype(float)

        penalty_accel = np.zeros(self.num_envs, dtype=np.float32)
        penalty_jerk = np.zeros(self.num_envs, dtype=np.float32)
        if (current_accel_penalty > 0 or current_jerk_penalty > 0) and self.dt > 0:
            current_accel = (arm_joint_vel_batch -
                             self.prev_joint_vel_batch) / self.dt
            if current_accel_penalty > 0:
                penalty_accel = -current_accel_penalty * \
                    np.sum(np.square(current_accel), axis=1)

            if current_jerk_penalty > 0:
                jerk = (current_accel - self.prev_joint_accel_batch) / self.dt
                penalty_jerk = -current_jerk_penalty * \
                    np.sum(np.square(jerk), axis=1)

            self.prev_joint_accel_batch = current_accel.copy()

        # Upright bonus for torque control to encourage fighting gravity
        upright_bonus = np.zeros(self.num_envs, dtype=np.float32)
        if current_upright_bonus > 0:
            upright_bonus = current_upright_bonus * \
                (ee_pos_batch[:, 2] > 0.1).astype(np.float32)

        # Total reward
        rewards_np = (reward_dist + reward_success + penalty_action_mag +
                      penalty_joint_limit + penalty_collision + penalty_joint_velocity +
                      penalty_ee_velocity + penalty_accel + penalty_jerk + upright_bonus)
        self.rew_buf = torch.from_numpy(rewards_np).to(self.device)
        self.episode_reward_buf += self.rew_buf

        # Update reset buffer
        self.reset_buf = torch.from_numpy(dones_np).to(self.device).long()

        # --- Populate the extras dictionary for logging ---
        self.extras.clear()
        self.extras["is_success"] = torch.from_numpy(
            terminated_success).to(self.device)
        self.extras["dist_to_target"] = torch.from_numpy(
            dist_to_target_batch).to(self.device)

        # Add individual reward components
        reward_components = {
            "distance": reward_dist, "success": reward_success, "action_penalty": penalty_action_mag,
            "joint_limit_penalty": penalty_joint_limit, "collision_penalty": penalty_collision,
            "joint_velocity_penalty": penalty_joint_velocity, "ee_velocity_penalty": penalty_ee_velocity,
            "accel_penalty": penalty_accel, "jerk_penalty": penalty_jerk, "upright_bonus": upright_bonus
        }
        for name, value in reward_components.items():
            self.extras[f"reward_components/{name}"] = torch.from_numpy(
                value).to(self.device)

        self.extras["debug/proximity_scaling_factor"] = torch.from_numpy(
            proximity_scaling_factor).to(self.device)

        # Add curriculum info for logging
        curriculum_components = {
            "overall_success_rate": self.current_success_rate,
            "threshold_value": self.threshold_curriculum.current_value,
            "joint_velocity_penalty_value": self.joint_velocity_penalty_curriculum.current_value,
            "ee_velocity_penalty_value": self.ee_velocity_penalty_curriculum.current_value,
            "action_penalty_value": self.action_penalty_curriculum.current_value,
            "accel_penalty_value": self.accel_penalty_curriculum.current_value,
            "jerk_penalty_value": self.jerk_penalty_curriculum.current_value,
            "upright_bonus_value": self.upright_bonus_curriculum.current_value
        }
        for name, value in curriculum_components.items():
            self.extras[f"curriculum/{name}"] = torch.tensor(
                value, device=self.device)

        # Log episode info when an episode is done
        for i in range(self.num_envs):
            if dones_np[i]:
                ep_info = {
                    'reward': self.episode_reward_buf[i].item(),
                    'length': self.episode_length_buf[i].item()
                }
                self.ep_infos.append(ep_info)

        # Update prev velocity for next step's acceleration penalty
        self.prev_joint_vel_batch = arm_joint_vel_batch.copy()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        if not self.scene or not self.franka_entity:
            raise RuntimeError("Env not initialized.")

        self.actions = actions.clone()

        # --- Apply actions based on the selected control mode ---
        actions_clipped = torch.clamp(self.actions, -1.0, 1.0)

        # --- Update action history buffer ---
        if self.num_actions_history > 0:
            # Flatten the current action
            current_action_flat = to_numpy(
                actions_clipped).reshape(self.num_envs, -1)
            # Roll the history buffer and insert the new action
            self.prev_actions_batch = np.roll(
                self.prev_actions_batch, shift=-self.FRANKA_NUM_ARM_JOINTS, axis=1)
            self.prev_actions_batch[:, -
                                    self.FRANKA_NUM_ARM_JOINTS:] = current_action_flat

        # --- Apply actions ---
        if self.control_mode == 'velocity':
            scaled_targets = actions_clipped * \
                torch.from_numpy(self.FRANKA_VEL_LIMIT).to(self.device)
            self.franka_entity.control_dofs_velocity(
                scaled_targets, self.franka_arm_dof_indices_local)
        elif self.control_mode == 'torque':
            scaled_torques = actions_clipped * \
                torch.from_numpy(self.FRANKA_TORQUE_LIMIT).to(self.device)
            self.franka_entity.control_dofs_force(
                scaled_torques, self.franka_arm_dof_indices_local)

        # Step simulation
        self.scene.step()
        if self._is_recording:
            self.video_capture_camera.render()

        self.episode_length_buf += 1

        # Compute rewards and observations
        self._calculate_rewards_and_dones()
        self._compute_observations()

        # Handle resets for environments that have finished their episode
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids_to_reset) > 0:
            # Target positions and sphere radii are updated inside reset_idx
            self.reset_idx(to_numpy(env_ids_to_reset))

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def close(self) -> None:
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

    def seed(self, seed=-1):
        if seed == -1:
            seed = np.random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]


# --- Main block for testing the environment ---
if __name__ == '__main__':
    print("--- FrankaShelfEnv (RSL-RL) Test Script ---")
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

    # --- Test Configuration ---
    # This setup mirrors the structure in train_agent.py for consistency.
    test_threshold_curriculum_cfg = {
        'start_value': 0.05, 'end_value': 0.005, 'start_metric_val': 0.4, 'end_metric_val': 0.8}
    test_velocity_penalty_curriculum_cfg = {
        'start_value': 0.0, 'end_value': 2.5, 'start_metric_val': 0.4, 'end_metric_val': 0.8}
    test_action_penalty_curriculum_cfg = {
        'start_value': 0.0001, 'end_value': 0.0005, 'start_metric_val': 0.2, 'end_metric_val': 0.7}
    test_accel_penalty_curriculum_cfg = {
        'start_value': 0.0, 'end_value': 0.0001, 'start_metric_val': 0.4, 'end_metric_val': 0.8}
    test_upright_bonus_curriculum_cfg = {
        'start_value': 0.2, 'end_value': 0.0, 'start_metric_val': 0.5, 'end_metric_val': 0.9}

    num_test_envs_for_run = 5
    print(f"\n\n--- TESTING WITH NUM_ENVS = {num_test_envs_for_run} ---")
    env = None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        env = FrankaShelfEnv(num_envs=num_test_envs_for_run,
                             control_mode='torque',  # or 'velocity'
                             threshold_curriculum_cfg=test_threshold_curriculum_cfg,
                             velocity_penalty_curriculum_cfg=test_velocity_penalty_curriculum_cfg,
                             action_penalty_curriculum_cfg=test_action_penalty_curriculum_cfg,
                             accel_penalty_curriculum_cfg=test_accel_penalty_curriculum_cfg,
                             upright_bonus_curriculum_cfg=test_upright_bonus_curriculum_cfg,
                             include_shelf=False,
                             randomize_shelf_config=False,
                             device=device,
                             render=True)

        print(
            f"FrankaShelfEnv created. Num Obs: {env.num_obs}, Num Act: {env.num_actions}")

        print("\nResetting environment...")
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        print(
            f"Target position for env 0: {env.target_position_world_batch[0]}")

        print("\nRunning a short loop with random actions...")
        for step_num in range(5000):
            # Manually advance the success rate for testing the curriculum
            if step_num > 0 and step_num % 100 == 0:
                # Simulate a successful episode to drive the curriculum forward
                env.success_buffer.append(True)

            actions = torch.rand(
                env.num_envs, env.num_actions, device=device) * 2 - 1
            obs, _, rewards, dones, infos = env.step(actions)
            if (step_num + 1) % 100 == 0:
                print(
                    f"Step {step_num+1}: Success Rate: {infos['curriculum/overall_success_rate']:.2f} | "
                    f"Upright Bonus: {infos['curriculum/upright_bonus_value']:.4f}")

        print("\nBasic test loop finished.")

    except Exception as e_runtime:
        print(
            f"ERROR during FrankaShelfEnv example usage (NUM_ENVS={num_test_envs_for_run}): {e_runtime}")
        traceback.print_exc()
    finally:
        if env:
            print("Closing environment...")
            env.close()

    if gs_initialized_in_main and hasattr(gs, 'shutdown') and callable(gs.shutdown):
        print("Shutting down Genesis post-testing.")
        gs.shutdown()
    print("--- FrankaShelfEnv Test Script Finished ---")
