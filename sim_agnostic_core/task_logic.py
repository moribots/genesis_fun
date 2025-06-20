"""
Contains the simulator-agnostic logic for defining and managing the task setup,
such as procedural generation of shelves and sampling of targets. This allows
the core task mechanics to be reused across different simulation backends.
"""
import random
import math
from typing import List, Any, Dict, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation


class FrankaTaskLogic:
    """
    Manages the procedural generation logic for the Franka shelf task.

    This class is responsible for:
    - Defining shelf geometry and configurations.
    - Sampling random shelf poses in the workspace.
    - Calculating the world-space poses of individual shelf components.
    - Sampling random target positions within the shelf opening.

    It is designed to be completely independent of the simulation engine.
    """
    # --- Shelf Geometry Constants ---
    DEFAULT_PLATE_WIDTH: float = 0.5
    DEFAULT_PLATE_DEPTH: float = 0.4
    DEFAULT_PLATE_THICKNESS: float = 0.02
    DEFAULT_WALL_THICKNESS: float = 0.02
    DEFAULT_ACTUAL_OPENING_HEIGHT: float = 0.25
    DEFAULT_WALL_COMPONENT_HEIGHT: float = DEFAULT_ACTUAL_OPENING_HEIGHT

    # Define component dimensions based on the constants
    COMPONENT_SIZE_PLATE: np.ndarray = np.array(
        [DEFAULT_PLATE_WIDTH, DEFAULT_PLATE_DEPTH, DEFAULT_PLATE_THICKNESS], dtype=np.float32)
    COMPONENT_SIZE_SIDE_WALL: np.ndarray = np.array(
        [DEFAULT_WALL_THICKNESS, DEFAULT_PLATE_DEPTH, DEFAULT_WALL_COMPONENT_HEIGHT], dtype=np.float32)
    COMPONENT_SIZE_BACK_WALL: np.ndarray = np.array(
        [DEFAULT_PLATE_WIDTH, DEFAULT_WALL_THICKNESS, DEFAULT_WALL_COMPONENT_HEIGHT], dtype=np.float32)

    # Ordered list of component sizes for assembly
    FIXED_COMPONENT_SIZES: List[np.ndarray] = [
        COMPONENT_SIZE_PLATE,  # Bottom plate
        COMPONENT_SIZE_PLATE,  # Top plate
        COMPONENT_SIZE_SIDE_WALL,  # Left wall
        COMPONENT_SIZE_SIDE_WALL,  # Right wall
        COMPONENT_SIZE_BACK_WALL  # Back wall
    ]
    SHELF_NUM_COMPONENTS: int = len(FIXED_COMPONENT_SIZES)

    def __init__(self, num_envs: int, randomize_shelf_config: bool, np_random: np.random.RandomState):
        """
        Initializes the task logic manager.

        Args:
            num_envs: The number of parallel environments.
            randomize_shelf_config: If True, shelf configurations are chosen
                                    randomly from a predefined set. If False,
                                    a default configuration is used.
            np_random: A NumPy random state object for reproducibility.
        """
        self.num_envs = num_envs
        self.randomize_shelf_config = randomize_shelf_config
        self.np_random = np_random

        # Define the available shelf placement strategies
        self._define_shelf_configurations()

        # Buffers to store the current state for each environment
        self.current_shelf_config_key_per_env: List[str] = [""] * self.num_envs
        self.current_shelf_instance_params_per_env: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)]

    def _define_shelf_configurations(self) -> None:
        """Defines the set of possible shelf configurations and their properties.

        Each configuration defines a region in the workspace where a shelf can be
        placed. This allows for creating different scenarios like high, low, or
        side reaches. A buffer zone around the robot base is enforced to prevent
        unsolvable placements.
        """
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

        # Validate that no configuration violates the central buffer zone
        buffer_half_width = 0.2
        for key, config_details in self.shelf_configurations.items():
            for x_range_tuple in config_details["base_pos_range_x"]:
                if not (x_range_tuple[1] <= -buffer_half_width or x_range_tuple[0] >= buffer_half_width):
                    raise ValueError(
                        f"Configuration '{key}' x_range {x_range_tuple} violates the buffer zone.")

    def _sample_shelf_assembly_poses(self, env_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the root position and orientation for the shelf assembly."""
        num_selected_envs = len(env_ids)
        origins = np.zeros((num_selected_envs, 3), dtype=np.float32)
        yaws = np.zeros(num_selected_envs, dtype=np.float32)

        for i, env_idx in enumerate(env_ids):
            # 1. Select a shelf configuration (e.g., 'high_center_reach')
            config_key = self.np_random.choice(
                self.shelf_config_keys_list) if self.randomize_shelf_config else self.default_shelf_config_key
            self.current_shelf_config_key_per_env[env_idx] = config_key
            config = self.shelf_configurations[config_key]

            # 2. Sample a base position within the configuration's range
            if self.randomize_shelf_config:
                chosen_x_sub_range = random.choice(config["base_pos_range_x"])
                origins[i, 0] = self.np_random.uniform(*chosen_x_sub_range)
                origins[i, 1] = self.np_random.uniform(
                    *config["base_pos_range_y"])
                origins[i, 2] = self.np_random.uniform(
                    *config["base_pos_range_z"])
            else:  # Use fixed, deterministic placement for debugging/testing
                origins[i, 0] = max(r[1] for r in config["base_pos_range_x"])
                origins[i, 1] = config["base_pos_range_y"][1]
                origins[i, 2] = config["base_pos_range_z"][1]

            # 3. Determine the shelf's yaw to face the robot's base
            if "center" in config["name"] or "forward" in config["name"]:
                yaws[i] = math.atan2(-origins[i, 1], -
                                     origins[i, 0]) - np.pi / 2.0
            elif "right" in config["name"]:
                yaws[i] = np.pi
            elif "left" in config["name"]:
                yaws[i] = 0.0
            else:
                yaws[i] = 0.0

            # 4. Store the instance parameters for later use (e.g., target sampling)
            self.current_shelf_instance_params_per_env[env_idx] = {
                "shelf_assembly_origin_world": origins[i].copy(),
                "shelf_assembly_yaw": yaws[i],
                "actual_opening_height": self.DEFAULT_ACTUAL_OPENING_HEIGHT,
                "internal_width": self.DEFAULT_PLATE_WIDTH - 2 * self.DEFAULT_WALL_THICKNESS,
                "internal_depth": self.DEFAULT_PLATE_DEPTH - self.DEFAULT_WALL_THICKNESS,
            }
        return origins, yaws

    def compute_shelf_component_poses(self, env_ids: Union[np.ndarray, List[int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes the world-space positions and orientations for all shelf parts.

        This function first samples a root pose for the entire shelf assembly,
        then calculates the final pose of each individual component (plates, walls)
        relative to that root pose.

        Args:
            env_ids: A list or array of environment indices to compute poses for.

        Returns:
            A tuple containing:
            - A list of batched component positions (num_components, num_envs, 3).
            - A list of batched component orientations (num_components, num_envs, 4).
        """
        if not len(env_ids):
            return [], []

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids

        # 1. Get the root position and orientation for the entire shelf assembly
        assembly_origins_world, assembly_yaws = self._sample_shelf_assembly_poses(
            env_ids_list)

        # Convert yaw angles to world-space quaternions (w, x, y, z)
        cos_yaw_half = np.cos(assembly_yaws / 2.0)
        sin_yaw_half = np.sin(assembly_yaws / 2.0)
        assembly_quats_wxyz = np.zeros(
            (len(env_ids_list), 4), dtype=np.float32)
        assembly_quats_wxyz[:, 0] = cos_yaw_half
        assembly_quats_wxyz[:, 3] = sin_yaw_half

        # 2. Define the local offsets of each component from the assembly origin
        b_plate_s, t_plate_s, s_wall_s, _, bk_wall_s = self.FIXED_COMPONENT_SIZES[:5]
        opening_h = self.DEFAULT_ACTUAL_OPENING_HEIGHT
        local_component_offsets = [
            np.array([0, 0, b_plate_s[2] / 2.0]),  # Bottom plate
            np.array([0, 0, b_plate_s[2] + opening_h +
                     t_plate_s[2] / 2.0]),  # Top plate
            np.array([-(b_plate_s[0]/2.0 - s_wall_s[0]/2.0), 0,
                     b_plate_s[2] + s_wall_s[2]/2.0]),  # Left wall
            np.array([(b_plate_s[0]/2.0 - s_wall_s[0]/2.0), 0,
                     b_plate_s[2] + s_wall_s[2]/2.0]),  # Right wall
            np.array([0, -(b_plate_s[1]/2.0 - bk_wall_s[1]/2.0),
                     b_plate_s[2] + bk_wall_s[2]/2.0])  # Back wall
        ]

        # 3. Transform local offsets to world-space for each component
        all_component_positions = []
        all_component_orientations = []
        for offset in local_component_offsets:
            # Rotate the local offset by the assembly's orientation
            assembly_quats_xyzw = assembly_quats_wxyz[:, [1, 2, 3, 0]]
            rotations = ScipyRotation.from_quat(assembly_quats_xyzw)
            rotated_offsets_world = rotations.apply(offset)

            # Add the rotated offset to the assembly origin to get the final world position
            final_positions = assembly_origins_world + rotated_offsets_world
            all_component_positions.append(final_positions)
            # All components share the same orientation as the parent assembly
            all_component_orientations.append(assembly_quats_wxyz)

        return all_component_positions, all_component_orientations

    def compute_target_positions(self, env_ids: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Computes new random target positions inside the shelf opening.

        This method uses the previously computed shelf instance parameters to
        sample a valid target point within the randomized shelf's internal volume.

        Args:
            env_ids: A list or array of environment indices to compute targets for.

        Returns:
            A NumPy array of shape (len(env_ids), 3) containing the new
            world-space target positions.
        """
        if not len(env_ids):
            return np.array([])

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids
        target_positions = np.zeros((len(env_ids_list), 3), dtype=np.float32)

        for i, env_idx in enumerate(env_ids_list):
            params = self.current_shelf_instance_params_per_env[env_idx]
            if not params:
                continue  # Skip if params haven't been generated for this env

            # 1. Sample a point within the shelf's internal opening frame (local coordinates)
            w, d, h = params["internal_width"], params["internal_depth"], params["actual_opening_height"]
            target_in_opening_frame = np.array([
                self.np_random.uniform(-w / 2 * 0.7, w / 2 * 0.7),
                # Bias towards front
                self.np_random.uniform(-d / 2 * 0.7, d / 2 * 0.3),
                self.np_random.uniform(-h / 2 * 0.8, h / 2 * 0.8)
            ], dtype=np.float32)

            # 2. Transform this local point into world coordinates
            assembly_origin = params["shelf_assembly_origin_world"]
            assembly_yaw = params["shelf_assembly_yaw"]

            # First, find the world-space center of the shelf's opening
            offset_to_opening_center = np.array(
                [0, 0, self.DEFAULT_PLATE_THICKNESS + h / 2.0], dtype=np.float32)
            rotation = ScipyRotation.from_euler('z', assembly_yaw)
            opening_center_world = assembly_origin + \
                rotation.apply(offset_to_opening_center)

            # Then, apply the same rotation to the sampled target point and add it to the opening center
            target_offset_world = rotation.apply(target_in_opening_frame)
            target_positions[i] = opening_center_world + target_offset_world

        return target_positions
