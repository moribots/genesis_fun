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
from collections import deque
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import torch

from sim_agnostic_core.curriculum_core import CurriculumConfig, LinearCurriculum
from sim_agnostic_core.env_core import BaseFrankaEnv
from sim_agnostic_core.task_logic import FrankaTaskLogic
from genesis_bridge.genesis_api import GenesisAPI, to_numpy


class GenesisFrankaEnv(BaseFrankaEnv):
    """
    A vectorized Franka Panda environment using the Genesis simulator.

    This class orchestrates the interaction between the high-level task logic
    (`FrankaTaskLogic`), the low-level simulator communication (`GenesisAPI`),
    and the reinforcement learning agent. It is responsible for stepping the
    simulation, computing observations and rewards, and resetting environments.

    Attributes:
        FRANKA_JOINT_NAMES (List[str]): Ordered list of all joint names.
        FRANKA_NUM_ARM_JOINTS (int): Number of arm joints (excluding fingers).
        ROBOT_EE_LINK_NAME (str): The name of the end-effector link in the MJCF.
        cfg (Dict): A dictionary holding the complete environment configuration.
        sim_api (GenesisAPI): The wrapper for the Genesis simulator.
        task_logic (FrankaTaskLogic): The manager for curriculum and task randomization.
    """
    # --- Robot Constants ---
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
    FRANKA_QPOS_RESET_NOISE_RANGES: np.ndarray = np.array(
        [(-1.0, 1.0), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)], dtype=np.float32)
    ROBOT_EE_LINK_NAME: str = "hand"

    def __init__(self, **kwargs):
        """
        Initializes the GenesisFrankaEnv.

        Args:
            **kwargs: A dictionary of configuration parameters. See `TrainConfig`
                      in `training_core.py` for expected keys.
        """
        # Initialize the abstract base class with core properties
        base_kwargs = {
            'num_envs': kwargs['num_envs'],
            'num_obs': kwargs['num_obs'],
            'num_actions': kwargs['num_actions'],
            'max_episode_length': kwargs['max_steps_per_episode'],
            'device': kwargs['device']
        }
        super().__init__(**base_kwargs)

        # --- Configuration and Seeding ---
        self.cfg = kwargs
        self.np_random = None
        self.seed(self.cfg.get('seed', 42))

        # --- Core Components ---
        self.control_mode = self.cfg.get('control_mode', 'velocity')
        self.num_actions_history = self.cfg.get('num_actions_history', 1)

        # --- Reward Coefficients ---
        self._load_reward_coefficients()

        # --- Initialize Helper Classes ---
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

        # --- Scene Creation ---
        self._create_simulation_scene()

        # --- Buffers ---
        self._init_buffers()

        # --- Simulator Handles ---
        self.franka_all_dof_indices_local = self.sim_api.get_franka_dof_indices(
            self.FRANKA_JOINT_NAMES)
        self.franka_arm_dof_indices_local = self.franka_all_dof_indices_local[
            :self.FRANKA_NUM_ARM_JOINTS]

        # --- Initial Reset ---
        self.reset()

    def _load_reward_coefficients(self):
        """Loads reward coefficients from the configuration dictionary."""
        self.k_dist_reward = self.cfg.get('k_dist_reward', 1.0)
        self.k_joint_limit_penalty = self.cfg.get(
            'k_joint_limit_penalty', 10.0)
        self.k_collision_penalty = self.cfg.get('k_collision_penalty', 100.0)
        self.success_reward = self.cfg.get('success_reward_val', 200.0)
        self.proximity_vel_penalty_max_scale = self.cfg.get(
            'proximity_vel_penalty_max_scale', 4.0)
        self.proximity_vel_penalty_dist_threshold = self.cfg.get(
            'proximity_vel_penalty_dist_threshold', 0.2)

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

        # Success rate tracking for curriculum
        self.min_episode_length_for_success_metric = self.cfg.get(
            'min_episode_length_for_success_metric', 10)
        self.success_buffer = deque(maxlen=100 * self.num_envs)
        self.current_success_rate = 0.0

    def _create_simulation_scene(self):
        """Delegates scene creation to the GenesisAPI."""
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

    def _init_buffers(self):
        """Initializes all required buffers for the environment."""
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

    def start_video_recording(self):
        """Starts video recording via the simulation API."""
        self.sim_api.start_video_recording()

    def stop_video_recording(self, file_path: str):
        """Stops video recording via the simulation API."""
        self.sim_api.stop_video_recording(file_path)

    def get_episode_infos(self) -> list:
        """Returns a copy of the episode infos and clears the internal list."""
        infos_to_return = self.ep_infos.copy()
        self.ep_infos.clear()
        return infos_to_return

    def _get_robot_state_batched(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves the batched robot state from the simulation API.

        Returns:
            A tuple containing batched NumPy arrays for:
            - arm_joint_pos_batch: Arm joint positions.
            - arm_joint_vel_batch: Arm joint velocities.
            - ee_pos_batch: End-effector positions.
            - ee_orient_quat_wxyz_batch: End-effector orientations.
            - ee_vel_batch: End-effector 6D velocities.
        """
        arm_joint_pos = self.sim_api.get_dof_positions(
            self.franka_arm_dof_indices_local)
        arm_joint_vel = self.sim_api.get_dof_velocities(
            self.franka_arm_dof_indices_local)
        ee_pos, ee_quat, ee_vel = self.sim_api.get_ee_state(
            self.ROBOT_EE_LINK_NAME)
        return arm_joint_pos, arm_joint_vel, ee_pos, ee_quat, ee_vel

    def _reset_task(self, env_ids: Union[np.ndarray, List[int]]):
        """
        Resets the task for specified environments.

        This involves generating new shelf configurations and target positions
        and applying them to the simulator.

        Args:
            env_ids: A list or array of environment indices to reset.
        """
        # 1. Compute new shelf component poses from the task logic
        component_positions, component_orientations = self.task_logic.compute_shelf_component_poses(
            env_ids)

        # 2. Apply these poses to the simulator
        if self.cfg.get('include_shelf', True):
            for i in range(self.task_logic.SHELF_NUM_COMPONENTS):
                self.sim_api.set_shelf_component_poses(
                    component_positions[i], component_orientations[i], i, env_ids)

        # 3. Compute new target positions and update the buffer
        new_targets = self.task_logic.compute_target_positions(env_ids)
        self.target_position_world_batch[env_ids] = new_targets

        # 4. Update visualizations in the simulator
        self._draw_target_spheres()

    def _draw_target_spheres(self):
        """Draws visualization spheres for targets in the simulator."""
        current_radius = self.threshold_curriculum.current_value
        is_rendering_or_recording = self.cfg.get(
            'render', False) or self.sim_api._is_recording

        if is_rendering_or_recording:
            if self.num_envs == 1:
                # For a single environment, draw a debug sphere
                points_to_draw = [self.target_position_world_batch[0].tolist()]
                self.sim_api.draw_debug_spheres(
                    points_to_draw, radius=current_radius, color=(0.0, 1.0, 0.0, 0.8))
            elif self.num_envs > 1:
                # For multiple environments, update the positions of the sphere entity type
                self.sim_api.set_target_sphere_positions(
                    self.target_position_world_batch)

    def _compute_observations(self) -> None:
        """
        Computes the full observation tensor from the current robot and task state.

        The observation includes: robot state (joint pos/vel, EE pos/quat),
        relative target position, EE velocity, and action history.
        """
        # Get current robot state
        arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz, ee_vel = self._get_robot_state_batched()

        # Concatenate robot state parts
        robot_state_flat = np.concatenate(
            [arm_qpos, arm_qvel, ee_pos, ee_quat_wxyz], axis=1, dtype=np.float32)

        # Calculate task-related state (relative target position)
        relative_target_pos = (
            self.target_position_world_batch - ee_pos).astype(np.float32)

        # Assemble the final observation tensor
        obs_parts = [robot_state_flat,
                     relative_target_pos, ee_vel.astype(np.float32)]
        if self.num_actions_history > 0:
            obs_parts.append(self.prev_actions_batch.astype(np.float32))

        obs_np = np.concatenate(obs_parts, axis=-1)
        self.obs_buf = torch.from_numpy(obs_np).to(self.device)
        self.extras['observations'] = {}

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Resets all environments to an initial state."""
        all_env_ids = np.arange(self.num_envs).tolist()
        self.reset_idx(all_env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: Union[np.ndarray, List[int]]):
        """
        Resets a specified set of environments by their indices.

        Args:
            env_ids: A list or array of environment indices to reset.
        """
        if not len(env_ids):
            return

        # 1. Reset robot state to a noisy initial configuration
        initial_qpos = np.tile(
            self.FRANKA_DEFAULT_INITIAL_QPOS, (len(env_ids), 1))
        low_noise, high_noise = self.FRANKA_QPOS_RESET_NOISE_RANGES[:,
                                                                    0], self.FRANKA_QPOS_RESET_NOISE_RANGES[:, 1]
        arm_noise = self.np_random.uniform(low_noise, high_noise, size=(
            len(env_ids), self.FRANKA_NUM_ARM_JOINTS))
        initial_qpos[:, :self.FRANKA_NUM_ARM_JOINTS] += arm_noise
        # Clip to ensure the reset state is within joint limits
        for i in range(self.FRANKA_NUM_ARM_JOINTS):
            initial_qpos[:, i] = np.clip(
                initial_qpos[:, i], self.FRANKA_QPOS_LOWER[i] + 0.01, self.FRANKA_QPOS_UPPER[i] - 0.01)
        initial_qvel = np.zeros(
            (len(env_ids), self.FRANKA_NUM_TOTAL_JOINTS), dtype=np.float32)
        self.sim_api.set_dof_state(
            initial_qpos, initial_qvel, self.franka_all_dof_indices_local, env_ids)

        # 2. Reset task-specific elements (shelves, targets)
        self._reset_task(env_ids)

        # 3. Reset custom buffers for these environments
        self.prev_joint_vel_batch[env_ids] = 0.0
        self.prev_joint_accel_batch[env_ids] = 0.0
        if self.num_actions_history > 0:
            self.prev_actions_batch[env_ids] = 0.0

        # 4. Reset RSL-RL buffers
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        self._compute_observations()
        return self.obs_buf

    def _update_curricula(self) -> Dict[str, float]:
        """Updates all curricula based on the current success rate and returns their values."""
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

        return {
            "success_threshold": self.threshold_curriculum.current_value,
            "joint_vel_penalty": self.joint_velocity_penalty_curriculum.current_value,
            "ee_vel_penalty": self.ee_velocity_penalty_curriculum.current_value,
            "action_penalty": self.action_penalty_curriculum.current_value,
            "accel_penalty": self.accel_penalty_curriculum.current_value,
            "jerk_penalty": self.jerk_penalty_curriculum.current_value,
            "upright_bonus": self.upright_bonus_curriculum.current_value,
        }

    def _compute_termination_and_success(self, dist_to_target: np.ndarray, success_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes termination conditions for all environments.

        Args:
            dist_to_target: Batch of distances to the target.
            success_threshold: The current success distance from the curriculum.

        Returns:
            A tuple containing boolean arrays for:
            - terminated: Any non-truncation termination.
            - terminated_success: Termination due to reaching the target.
            - dones: Any termination, including truncation.
        """
        terminated_success = dist_to_target < success_threshold
        terminated_collision = self.sim_api.get_collisions()
        terminated = np.logical_or(terminated_collision, terminated_success)
        truncated = self.episode_length_buf.cpu().numpy() >= self.max_episode_length
        dones = np.logical_or(terminated, truncated)

        # Update success buffer for curriculum
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            if self.episode_length_buf[idx] >= self.min_episode_length_for_success_metric:
                self.success_buffer.append(terminated_success[idx])

        return terminated_collision, terminated_success, dones

    def _compute_reward_components(self, state: Dict, curriculum_vals: Dict, terminated_collision: np.ndarray, terminated_success: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates all individual reward and penalty components.

        Args:
            state: A dictionary containing current state arrays (qpos, qvel, etc.).
            curriculum_vals: A dictionary of current values from all curricula.
            terminated_collision: Boolean array indicating collision terminations.
            terminated_success: Boolean array indicating success terminations.

        Returns:
            A dictionary where keys are reward component names and values are batch arrays.
        """
        # --- Unpack state and curriculum values ---
        arm_qpos, arm_qvel, ee_pos, ee_vel, actions = state['arm_qpos'], state[
            'arm_qvel'], state['ee_pos'], state['ee_vel'], state['actions']
        dist_to_target = state['dist_to_target']
        k_jp, k_eevp, k_ap, k_accp, k_jpk, k_ub = (curriculum_vals['joint_vel_penalty'], curriculum_vals['ee_vel_penalty'],
                                                   curriculum_vals['action_penalty'], curriculum_vals['accel_penalty'],
                                                   curriculum_vals['jerk_penalty'], curriculum_vals['upright_bonus'])

        # --- Base Rewards and Penalties ---
        rewards = {}
        rewards["distance"] = -self.k_dist_reward * dist_to_target
        rewards["success"] = self.success_reward * \
            terminated_success.astype(float)
        rewards["action_penalty"] = -k_ap * np.sum(np.square(actions), axis=1)
        out_of_bounds = (arm_qpos < self.FRANKA_QPOS_LOWER) | (
            arm_qpos > self.FRANKA_QPOS_UPPER)
        rewards["joint_limit_penalty"] = - \
            self.k_joint_limit_penalty * np.sum(out_of_bounds, axis=1)
        rewards["collision_penalty"] = -self.k_collision_penalty * \
            terminated_collision.astype(float)

        # --- Velocity Penalties with Proximity Scaling ---
        proximity_scaling = np.ones(self.num_envs, dtype=np.float32)
        if self.proximity_vel_penalty_dist_threshold > 0:
            close_mask = dist_to_target < self.proximity_vel_penalty_dist_threshold
            norm_dist_inv = 1.0 - \
                (dist_to_target[close_mask] /
                 self.proximity_vel_penalty_dist_threshold)
            proximity_scaling[close_mask] += (
                self.proximity_vel_penalty_max_scale - 1.0) * norm_dist_inv
        rewards["joint_velocity_penalty"] = -k_jp * \
            np.linalg.norm(arm_qvel, axis=1) * proximity_scaling
        rewards["ee_velocity_penalty"] = -k_eevp * \
            np.linalg.norm(ee_vel, axis=1) * proximity_scaling

        # --- Acceleration and Jerk Penalties ---
        rewards["accel_penalty"] = np.zeros(self.num_envs, dtype=np.float32)
        rewards["jerk_penalty"] = np.zeros(self.num_envs, dtype=np.float32)
        if (k_accp > 0 or k_jpk > 0) and self.cfg['dt'] > 0:
            current_accel = (
                arm_qvel - self.prev_joint_vel_batch) / self.cfg['dt']
            if k_accp > 0:
                rewards["accel_penalty"] = -k_accp * \
                    np.sum(np.square(current_accel), axis=1)
            if k_jpk > 0:
                jerk = (current_accel - self.prev_joint_accel_batch) / \
                    self.cfg['dt']
                rewards["jerk_penalty"] = -k_jpk * \
                    np.sum(np.square(jerk), axis=1)
            self.prev_joint_accel_batch = current_accel.copy()

        # --- Upright Bonus ---
        rewards["upright_bonus"] = k_ub * (ee_pos[:, 2] > 0.1).astype(
            np.float32) if k_ub > 0 else np.zeros(self.num_envs, dtype=np.float32)

        self.extras["debug/proximity_scaling_factor"] = torch.from_numpy(
            proximity_scaling).to(self.device)
        return rewards

    def _calculate_rewards_and_dones(self) -> None:
        """Calculates rewards and dones for the current step."""
        # 1. Get current state from simulation
        actions_np = to_numpy(self.actions)
        arm_qpos, arm_qvel, ee_pos, _, ee_vel = self._get_robot_state_batched()
        dist_to_target = np.linalg.norm(
            ee_pos - self.target_position_world_batch, axis=1)
        state_dict = {
            'arm_qpos': arm_qpos, 'arm_qvel': arm_qvel, 'ee_pos': ee_pos, 'ee_vel': ee_vel,
            'actions': actions_np, 'dist_to_target': dist_to_target
        }

        # 2. Update curricula and get current penalty/reward weights
        curriculum_vals = self._update_curricula()

        # 3. Determine termination conditions
        terminated_collision, terminated_success, dones_np = self._compute_termination_and_success(
            dist_to_target, curriculum_vals['success_threshold'])

        # 4. Calculate all reward components
        reward_components = self._compute_reward_components(
            state_dict, curriculum_vals, terminated_collision, terminated_success)

        # 5. Sum rewards and update buffers
        total_rewards_np = sum(reward_components.values())
        self.rew_buf = torch.from_numpy(total_rewards_np).to(self.device)
        self.episode_reward_buf += self.rew_buf
        self.reset_buf = torch.from_numpy(dones_np).to(self.device).long()

        # 6. Populate `extras` dictionary for logging
        self._populate_extras_for_logging(
            terminated_success, dist_to_target, reward_components, curriculum_vals)

        # 7. Handle episode-end logging
        for i in np.where(dones_np)[0]:
            self.ep_infos.append({'reward': self.episode_reward_buf[i].item(
            ), 'length': self.episode_length_buf[i].item()})

        # 8. Update state history for next step's calculations
        self.prev_joint_vel_batch = arm_qvel.copy()

    def _populate_extras_for_logging(self, terminated_success, dist_to_target, reward_components, curriculum_vals):
        """Populates the `extras` dictionary with data for the logger."""
        self.extras.clear()
        self.extras["is_success"] = torch.from_numpy(
            terminated_success).to(self.device)
        self.extras["dist_to_target"] = torch.from_numpy(
            dist_to_target).to(self.device)

        for name, value in reward_components.items():
            self.extras[f"reward_components/{name}"] = torch.from_numpy(
                value).to(self.device)

        curriculum_log_map = {
            "overall_success_rate": self.current_success_rate,
            "threshold_value": curriculum_vals["success_threshold"],
            "joint_velocity_penalty_value": curriculum_vals["joint_vel_penalty"],
            "ee_velocity_penalty_value": curriculum_vals["ee_vel_penalty"],
            "action_penalty_value": curriculum_vals["action_penalty"],
            "accel_penalty_value": curriculum_vals["accel_penalty"],
            "jerk_penalty_value": curriculum_vals["jerk_penalty"],
            "upright_bonus_value": curriculum_vals["upright_bonus"],
        }
        for name, value in curriculum_log_map.items():
            self.extras[f"curriculum/{name}"] = torch.tensor(
                value, device=self.device)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """Performs a single environment step."""
        self.actions = actions.clone()
        actions_clipped = torch.clamp(self.actions, -1.0, 1.0)

        # Update action history buffer for observation
        if self.num_actions_history > 0:
            current_action_flat = to_numpy(
                actions_clipped).reshape(self.num_envs, -1)
            self.prev_actions_batch = np.roll(
                self.prev_actions_batch, shift=-self.FRANKA_NUM_ARM_JOINTS, axis=1)
            self.prev_actions_batch[:, -
                                    self.FRANKA_NUM_ARM_JOINTS:] = current_action_flat

        # Apply actions to the simulator based on control mode
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

        # Step the physics simulation
        self.sim_api.step()
        self.episode_length_buf += 1

        # Post-physics calculations
        self._calculate_rewards_and_dones()
        self._compute_observations()

        # Handle environment resets for those that are done
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids_to_reset) > 0:
            self.reset_idx(to_numpy(env_ids_to_reset).tolist())

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def close(self) -> None:
        """Cleans up simulator resources."""
        self.sim_api.close()

    def seed(self, seed: int = -1) -> List[int]:
        """Sets the random seed for the environment and its components."""
        if seed == -1:
            seed = np.random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random = np.random.RandomState(seed)
        # Also seed the task logic's random state
        if hasattr(self, 'task_logic') and self.task_logic:
            self.task_logic.np_random = self.np_random
        return [seed]
