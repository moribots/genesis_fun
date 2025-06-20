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
    Manages the logic for the Franka shelf and target randomization task.
    This class is independent of the simulation engine.
    """
    # Shelf Constants
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
    FIXED_COMPONENT_SIZES: List[np.ndarray] = [COMPONENT_SIZE_PLATE, COMPONENT_SIZE_PLATE,
                                               COMPONENT_SIZE_SIDE_WALL, COMPONENT_SIZE_SIDE_WALL, COMPONENT_SIZE_BACK_WALL]
    SHELF_NUM_COMPONENTS: int = len(FIXED_COMPONENT_SIZES)

    def __init__(self, num_envs: int, randomize_shelf_config: bool, np_random: np.random.RandomState):
        """
        Initializes the task logic manager.

        Args:
            num_envs: The number of parallel environments.
            randomize_shelf_config: Whether to randomize the shelf configuration.
            np_random: A NumPy random state object for reproducibility.
        """
        self.num_envs = num_envs
        self.randomize_shelf_config = randomize_shelf_config
        self.np_random = np_random
        self._define_shelf_configurations()

        self.current_shelf_config_key_per_env: List[str] = [""] * self.num_envs
        self.current_shelf_instance_params_per_env: List[Dict[str, Any]] = [
            {} for _ in range(self.num_envs)]

    def _define_shelf_configurations(self) -> None:
        """Defines the set of possible shelf configurations and their properties."""
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
                        f"Configuration '{key}' x_range {x_range_tuple} violates the buffer zone.")

    def compute_shelf_component_poses(self, env_ids: Union[np.ndarray, List[int]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Computes the world-space positions and orientations for all shelf components
        for the specified environments.
        """
        if not len(env_ids):
            return [], []

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids

        shelf_assembly_origins_world_batch = np.zeros(
            (len(env_ids_list), 3), dtype=np.float32)
        shelf_assembly_yaws_batch = np.zeros(
            len(env_ids_list), dtype=np.float32)

        for i, env_idx in enumerate(env_ids_list):
            config_key = self.np_random.choice(
                self.shelf_config_keys_list) if self.randomize_shelf_config else self.default_shelf_config_key
            self.current_shelf_config_key_per_env[env_idx] = config_key
            config = self.shelf_configurations[config_key]

            if self.randomize_shelf_config:
                x_ranges_for_config = config["base_pos_range_x"]
                chosen_x_sub_range = random.choice(x_ranges_for_config)
                shelf_assembly_origins_world_batch[i, 0] = self.np_random.uniform(
                    *chosen_x_sub_range)
                shelf_assembly_origins_world_batch[i, 1] = self.np_random.uniform(
                    *config["base_pos_range_y"])
                shelf_assembly_origins_world_batch[i, 2] = self.np_random.uniform(
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
            np.array([0, 0, bottom_plate_s[2]/2.0]),
            np.array([0, 0, bottom_plate_s[2] +
                     actual_opening_h+top_plate_s[2]/2.0]),
            np.array([-(bottom_plate_s[0]/2.0-side_wall_s[0]/2.0),
                     0, bottom_plate_s[2]+side_wall_s[2]/2.0]),
            np.array([(bottom_plate_s[0]/2.0-side_wall_s[0]/2.0),
                     0, bottom_plate_s[2]+side_wall_s[2]/2.0]),
            np.array([0, -(bottom_plate_s[1]/2.0-back_wall_s[1]/2.0),
                     bottom_plate_s[2]+back_wall_s[2]/2.0])
        ]

        all_component_positions = []
        all_component_orientations = []

        for comp_idx in range(self.SHELF_NUM_COMPONENTS):
            batched_final_comp_pos_world = np.zeros(
                (len(env_ids_list), 3), dtype=np.float32)
            for i in range(len(env_ids_list)):
                assembly_quat_xyzw_i = shelf_assembly_world_quats_batch_wxyz[i, [
                    1, 2, 3, 0]]
                R_shelf_assembly_yaw_i = ScipyRotation.from_quat(
                    assembly_quat_xyzw_i).as_matrix()
                rotated_offset_world_i = R_shelf_assembly_yaw_i @ local_component_offsets[comp_idx]
                batched_final_comp_pos_world[i] = shelf_assembly_origins_world_batch[i] + \
                    rotated_offset_world_i
            all_component_positions.append(batched_final_comp_pos_world)
            all_component_orientations.append(
                shelf_assembly_world_quats_batch_wxyz)

        return all_component_positions, all_component_orientations

    def compute_target_positions(self, env_ids: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Computes new random target positions for the specified environments based
        on their current shelf configurations.
        """
        if not len(env_ids):
            return np.array([])

        env_ids_list = env_ids.tolist() if isinstance(env_ids, np.ndarray) else env_ids
        target_positions = np.zeros((len(env_ids_list), 3), dtype=np.float32)

        for i, env_idx in enumerate(env_ids_list):
            params = self.current_shelf_instance_params_per_env[env_idx]
            if not params:
                continue

            open_w, open_d, open_h = params["internal_width"], params["internal_depth"], params["actual_opening_height"]
            target_in_shelf_opening_frame = np.array([
                self.np_random.uniform(-open_w/2*0.7, open_w/2*0.7),
                self.np_random.uniform(-open_d/2*0.7, open_d/2*0.3),
                self.np_random.uniform(-open_h/2*0.8, open_h/2*0.8)
            ], dtype=np.float32)

            shelf_assembly_origin_world = params["shelf_assembly_origin_world"]
            shelf_assembly_yaw = params["shelf_assembly_yaw"]
            offset_to_opening_center_local = np.array(
                [0, 0, self.DEFAULT_PLATE_THICKNESS + open_h/2.0], dtype=np.float32)
            shelf_assembly_quat_world_wxyz = np.array([np.cos(
                shelf_assembly_yaw/2.0), 0, 0, np.sin(shelf_assembly_yaw/2.0)], dtype=np.float32)
            shelf_assembly_quat_world_xyzw = shelf_assembly_quat_world_wxyz[[
                1, 2, 3, 0]]

            R_shelf_yaw_world = ScipyRotation.from_quat(
                shelf_assembly_quat_world_xyzw).as_matrix()
            opening_center_world = shelf_assembly_origin_world + \
                (R_shelf_yaw_world @ offset_to_opening_center_local)
            target_offset_world_rotated = R_shelf_yaw_world @ target_in_shelf_opening_frame
            target_positions[i] = opening_center_world + \
                target_offset_world_rotated

        return target_positions
