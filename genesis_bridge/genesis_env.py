"""
Defines the GenesisFrankaEnv, a concrete implementation of the vectorized
reinforcement learning environment for a Franka Emika Panda robot that uses
the Genesis simulation engine for physics and rendering.

This class inherits from the simulator-agnostic `BaseFrankaEnv` and uses the
`GenesisAPI` to interact with the simulation. It handles all logic related to
robot state, rewards, and environment randomization that is specific to the
Genesis implementation.
"""
import random
import math
from typing import List, Any, Optional, Union, Dict, Tuple
from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from sim_agnostic_core.env_core import BaseFrankaEnv
from sim_agnostic_core.curriculum_core import LinearCurriculum, CurriculumConfig
from sim_agnostic_core.task_logic import FrankaTaskLogic
from genesis_bridge.genesis_api import GenesisAPI, to_numpy


class GenesisFrankaEnv(BaseFrankaEnv):
    """
    A vectorized environment for a Franka Emika Panda robot, using the Genesis
    simulator and adapted for RSL-RL. This class acts as an orchestrator,
    connecting the simulator-agnostic task logic with the Genesis API.
    """
    # Robot Constants
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
    FRANKA_QPOS_RESET_NOISE_RANGES: np.ndarray = np.array(
        [(-1.0, 1.0), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)], dtype=np.float32)

    def __init__(self, **kwargs):
        # Initialize the abstract base class
        base_kwargs = {
            'num_envs': kwargs['num_envs'],
            'num_obs': kwargs['num_obs'],
            'num_actions': kwargs['num_actions'],
            'max_episode_length': kwargs['max_steps_per_episode'],
            'device': kwargs['device']
        }
        super().__init__(**base_kwargs)

        # Store all other configuration parameters
        self.cfg = kwargs
        self.np_random = None
        self.seed(self.cfg.get('seed', 42))

        # Core components
        self.control_mode = self.cfg.get('control_mode', 'velocity')
        self.num_actions_history = self.cfg.get('num_actions_history', 1)

        # Reward coefficients
        self.k_dist_reward = self.cfg.get('k_dist_reward', 1.0)
        self.k_joint_limit_penalty = self.cfg.get(
            'k_joint_limit_penalty', 10.0)
        self.k_collision_penalty = self.cfg.get('k_collision_penalty', 100.0)
        self.success_reward = self.cfg.get('success_reward_val', 200.0)
        self.proximity_vel_penalty_max_scale = self.cfg.get(
            'proximity_vel_penalty_max_scale', 4.0)
        self.proximity_vel_penalty_dist_threshold = self.cfg.get(
            'proximity_vel_penalty_dist_threshold', 0.2)

        # Initialize helper classes
        self._init_curriculum()
        self.task_logic = FrankaTaskLogic(
            num_envs=self.num_envs,
            randomize_shelf_config=self.cfg.get(
                'randomize_shelf_config', True),
            np_random=self.np_random
        )
        self.sim_api = GenesisAPI(dt=self.cfg['dt'], render=self.cfg.get(
            'render', False), device=self.device)
        self.sim_api.initialize()

        # Create the scene in the simulator
        self.sim_api.create_scene(
            num_envs=self.num_envs,
            env_spacing=self.cfg.get('env_spacing', (1.5, 1.5)),
            video_camera_params={
                'res': self.cfg['video_res'], 'pos': self.cfg['video_camera_pos'],
                'lookat': self.cfg['video_camera_lookat'], 'fov': self.cfg['video_camera_fov']
            },
            franka_xml_path=self.cfg['franka_xml_path'],
            shelf_component_sizes=self.task_logic.FIXED_COMPONENT_SIZES,
            include_shelf=self.cfg.get('include_shelf', True)
        )

        # Buffers
        self.episode_reward_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.ep_infos = []
        self.target_position_world_batch = np.zeros(
            (self.num_envs, 3), dtype=np.float32)
        self.prev_joint_vel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)
        self.prev_joint_accel_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS), dtype=np.float32)
        self.prev_actions_batch = np.zeros(
            (self.num_envs, self.FRANKA_NUM_ARM_JOINTS * self.num_actions_history), dtype=np.float32)

        # Get DOF indices from simulator
        self.franka_all_dof_indices_local = self.sim_api.get_franka_dof_indices(
            self.FRANKA_JOINT_NAMES)
        self.franka_arm_dof_indices_local = self.franka_all_dof_indices_local[
            :self.FRANKA_NUM_ARM_JOINTS]

        self.reset()

    def _init_curriculum(self):
        """Initializes all curriculum components from the config."""
        self.threshold_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['threshold_curriculum']))
        self.joint_velocity_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['joint_velocity_penalty_curriculum']))
        self.ee_velocity_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['ee_velocity_penalty_curriculum']))
        self.action_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['action_penalty_curriculum']))
        self.accel_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['accel_penalty_curriculum']))
        self.jerk_penalty_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['jerk_penalty_curriculum']))
        self.upright_bonus_curriculum = LinearCurriculum(
            CurriculumConfig(**self.cfg['upright_bonus_curriculum']))
        self.min_episode_length_for_success_metric = self.cfg.get(
            'min_episode_length_for_success_metric', 10)
        self.success_buffer = deque(maxlen=100 * self.num_envs)
        self.current_success_rate = 0.0

    def start_video_recording(self):
        self.sim_api.start_video_recording()

    def stop_video_recording(self, file_path: str):
        self.sim_api.stop_video_recording(file_path)

    def get_episode_infos(self):
        infos_to_return = self.ep_infos.copy()
        self.ep_infos.clear()
        return infos_to_return

    def _get_robot_state_parts_batched(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gets robot state by calling the simulation API."""
        arm_joint_pos_batch = self.sim_api.get_dof_positions(
            self.franka_arm_dof_indices_local)
        arm_joint_vel_batch = self.sim_api.get_dof_velocities(
            self.franka_arm_dof_indices_local)
        ee_pos_batch, ee_orient_quat_wxyz_batch, ee_vel_batch = self.sim_api.get_ee_state(
            self.ROBOT_EE_LINK_NAME)
        return arm_joint_pos_batch, arm_joint_vel_batch, ee_pos_batch, ee_orient_quat_wxyz_batch, ee_vel_batch

    def _reset_task(self, env_ids: Union[np.ndarray, List[int]]):
        """Resets the task by generating new shelf and target poses and sending them to the simulator."""
        # 1. Compute new shelf component poses from the task logic
        component_positions, component_orientations = self.task_logic.compute_shelf_component_poses(
            env_ids)

        # 2. Apply these poses to the simulator
        if self.cfg.get('include_shelf', True):
            for i in range(self.task_logic.SHELF_NUM_COMPONENTS):
                self.sim_api.set_shelf_component_poses(
                    component_positions[i],
                    component_orientations[i],
                    i,
                    env_ids
                )

        # 3. Compute new target positions from the task logic
        new_targets = self.task_logic.compute_target_positions(env_ids)
        self.target_position_world_batch[env_ids] = new_targets

        # 4. Draw visualizations in the simulator
        self._draw_target_spheres()

    def _draw_target_spheres(self):
        """Draws visualization spheres in the simulator."""
        current_radius = self.threshold_curriculum.current_value
        if (self.cfg.get('render', False) or (hasattr(self.sim_api, '_is_recording') and self.sim_api._is_recording)):
            if self.num_envs == 1:
                points_to_draw = [self.target_position_world_batch[0].tolist()]
                self.sim_api.draw_debug_spheres(
                    points_to_draw, radius=current_radius, color=(0.0, 1.0, 0.0, 0.8))
            elif self.num_envs > 1:
                self.sim_api.set_target_sphere_positions(
                    self.target_position_world_batch)

    def _compute_observations(self) -> None:
        """Computes the observation tensor from the robot state and task state."""
        arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz, ee_vel = self._get_robot_state_parts_batched()
        robot_state_flat_batch = np.concatenate(
            [arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz], axis=1).astype(np.float32)
        relative_target_pos_batch = (
            self.target_position_world_batch - ee_pos).astype(np.float32)
        obs_parts = [robot_state_flat_batch,
                     relative_target_pos_batch, ee_vel.astype(np.float32)]
        if self.num_actions_history > 0:
            obs_parts.append(self.prev_actions_batch.astype(np.float32))
        obs_np = np.concatenate(obs_parts, axis=-1)
        self.obs_buf = torch.from_numpy(obs_np).to(self.device)
        self.extras['observations'] = {}

    def reset(self):
        """Resets all environments."""
        all_env_ids = np.arange(self.num_envs).tolist()
        self.reset_idx(all_env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: Union[np.ndarray, List[int]]):
        """Resets specified environments."""
        if not len(env_ids):
            return

        # Reset robot state
        initial_qpos = np.tile(
            self.FRANKA_DEFAULT_INITIAL_QPOS, (len(env_ids), 1))
        low_noise, high_noise = self.FRANKA_QPOS_RESET_NOISE_RANGES[:,
                                                                    0], self.FRANKA_QPOS_RESET_NOISE_RANGES[:, 1]
        arm_noise = self.np_random.uniform(low_noise, high_noise, size=(
            len(env_ids), self.FRANKA_NUM_ARM_JOINTS))
        initial_qpos[:, :self.FRANKA_NUM_ARM_JOINTS] += arm_noise
        for i in range(self.FRANKA_NUM_ARM_JOINTS):
            initial_qpos[:, i] = np.clip(
                initial_qpos[:, i], self.FRANKA_QPOS_LOWER[i] + 0.01, self.FRANKA_QPOS_UPPER[i] - 0.01)
        initial_qvel = np.zeros(
            (len(env_ids), self.FRANKA_NUM_TOTAL_JOINTS), dtype=np.float32)
        self.sim_api.set_dof_state(
            initial_qpos, initial_qvel, self.franka_all_dof_indices_local, env_ids)

        # Reset task-specific elements (shelves, targets)
        self._reset_task(env_ids)

        # Reset custom buffers
        self.prev_joint_vel_batch[env_ids] = 0.0
        self.prev_joint_accel_batch[env_ids] = 0.0
        if self.num_actions_history > 0:
            self.prev_actions_batch[env_ids] = 0.0

        # Reset RSL-RL buffers
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self._compute_observations()
        return self.obs_buf

    def _calculate_rewards_and_dones(self) -> None:
        """Calculates rewards and dones for the current step."""
        actions_clipped_batch = to_numpy(self.actions)
        arm_joint_pos_batch, arm_joint_vel_batch, ee_pos_batch, _, ee_vel_batch = self._get_robot_state_parts_batched()
        dist_to_target_batch = np.linalg.norm(
            ee_pos_batch - self.target_position_world_batch, axis=1)

        # --- Curriculum Update ---
        if len(self.success_buffer) > 0:
            self.current_success_rate = np.mean(list(self.success_buffer))

        self.threshold_curriculum.update(self.current_success_rate)
        self.joint_velocity_penalty_curriculum.update(
            self.current_success_rate)
        self.ee_velocity_penalty_curriculum.update(self.current_success_rate)
        self.action_penalty_curriculum.update(self.current_success_rate)
        self.accel_penalty_curriculum.update(self.current_success_rate)
        self.jerk_penalty_curriculum.update(self.current_success_rate)
        self.upright_bonus_curriculum.update(self.current_success_rate)

        current_success_threshold = self.threshold_curriculum.current_value
        current_joint_vel_penalty = self.joint_velocity_penalty_curriculum.current_value
        current_ee_vel_penalty = self.ee_velocity_penalty_curriculum.current_value
        current_action_penalty = self.action_penalty_curriculum.current_value
        current_accel_penalty = self.accel_penalty_curriculum.current_value
        current_jerk_penalty = self.jerk_penalty_curriculum.current_value
        current_upright_bonus = self.upright_bonus_curriculum.current_value

        # --- Dones ---
        terminated_success = dist_to_target_batch < current_success_threshold
        terminated_collision = self.sim_api.get_collisions()
        terminated = np.logical_or(terminated_collision, terminated_success)
        truncated = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        dones_np = np.logical_or(terminated, truncated)
        done_indices = np.where(dones_np)[0]
        for idx in done_indices:
            if self.episode_length_buf[idx] >= self.min_episode_length_for_success_metric:
                self.success_buffer.append(terminated_success[idx])

        # --- Reward Components ---
        reward_dist = -self.k_dist_reward * dist_to_target_batch
        penalty_action_mag = -current_action_penalty * \
            np.sum(np.square(actions_clipped_batch), axis=1)
        out_of_bounds = (arm_joint_pos_batch < self.FRANKA_QPOS_LOWER) | (
            arm_joint_pos_batch > self.FRANKA_QPOS_UPPER)
        penalty_joint_limit = -self.k_joint_limit_penalty * \
            np.sum(out_of_bounds, axis=1)
        penalty_collision = -self.k_collision_penalty * \
            terminated_collision.astype(float)

        proximity_scaling_factor = np.ones(self.num_envs, dtype=np.float32)
        if self.proximity_vel_penalty_dist_threshold > 0:
            close_mask = dist_to_target_batch < self.proximity_vel_penalty_dist_threshold
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
        if (current_accel_penalty > 0 or current_jerk_penalty > 0) and self.cfg['dt'] > 0:
            current_accel = (arm_joint_vel_batch -
                             self.prev_joint_vel_batch) / self.cfg['dt']
            if current_accel_penalty > 0:
                penalty_accel = -current_accel_penalty * \
                    np.sum(np.square(current_accel), axis=1)
            if current_jerk_penalty > 0:
                jerk = (current_accel - self.prev_joint_accel_batch) / \
                    self.cfg['dt']
                penalty_jerk = -current_jerk_penalty * \
                    np.sum(np.square(jerk), axis=1)
            self.prev_joint_accel_batch = current_accel.copy()

        upright_bonus = np.zeros(self.num_envs, dtype=np.float32)
        if current_upright_bonus > 0:
            upright_bonus = current_upright_bonus * \
                (ee_pos_batch[:, 2] > 0.1).astype(np.float32)

        rewards_np = (reward_dist + reward_success + penalty_action_mag +
                      penalty_joint_limit + penalty_collision + penalty_joint_velocity +
                      penalty_ee_velocity + penalty_accel + penalty_jerk + upright_bonus)

        self.rew_buf = torch.from_numpy(rewards_np).to(self.device)
        self.episode_reward_buf += self.rew_buf
        self.reset_buf = torch.from_numpy(dones_np).to(self.device).long()

        # --- Populate the extras dictionary for logging ---
        self.extras.clear()
        self.extras["is_success"] = torch.from_numpy(
            terminated_success).to(self.device)
        self.extras["dist_to_target"] = torch.from_numpy(
            dist_to_target_batch).to(self.device)

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

        for i in range(self.num_envs):
            if dones_np[i]:
                ep_info = {'reward': self.episode_reward_buf[i].item(
                ), 'length': self.episode_length_buf[i].item()}
                self.ep_infos.append(ep_info)

        self.prev_joint_vel_batch = arm_joint_vel_batch.copy()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Performs a single environment step."""
        self.actions = actions.clone()
        actions_clipped = torch.clamp(self.actions, -1.0, 1.0)

        # Update action history buffer
        if self.num_actions_history > 0:
            current_action_flat = to_numpy(
                actions_clipped).reshape(self.num_envs, -1)
            self.prev_actions_batch = np.roll(
                self.prev_actions_batch, shift=-self.FRANKA_NUM_ARM_JOINTS, axis=1)
            self.prev_actions_batch[:, -
                                    self.FRANKA_NUM_ARM_JOINTS:] = current_action_flat

        # Apply actions to the simulator
        if self.control_mode == 'velocity':
            scaled_targets = actions_clipped * \
                torch.from_numpy(self.FRANKA_VEL_LIMIT).to(self.device)
            self.sim_api.apply_velocity_control(
                scaled_targets, self.franka_arm_dof_indices_local)
        elif self.control_mode == 'torque':
            scaled_torques = actions_clipped * \
                torch.from_numpy(self.FRANKA_TORQUE_LIMIT).to(self.device)
            self.sim_api.apply_torque_control(
                scaled_torques, self.franka_arm_dof_indices_local)

        # Step the simulation
        self.sim_api.step()
        self.episode_length_buf += 1

        # Post-physics calculations
        self._calculate_rewards_and_dones()
        self._compute_observations()

        # Handle resets
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids_to_reset) > 0:
            self.reset_idx(to_numpy(env_ids_to_reset).tolist())

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def close(self) -> None:
        """Cleans up simulator resources."""
        self.sim_api.close()

    def seed(self, seed=-1):
        """Sets the random seed for the environment."""
        if seed == -1:
            seed = np.random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random = np.random.RandomState(seed)
        return [seed]
