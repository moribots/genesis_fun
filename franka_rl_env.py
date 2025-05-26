import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch  # For type checking and tensor operations if Genesis returns PyTorch tensors
import random
import math

# Import Genesis
import genesis as gs  # Assuming 'genesis' is the correct import name

# Initialize Genesis Simulator
# gs.init(backend=gs.cpu) # Call this once globally at the start of your main script.


def to_numpy(data):
    """
    Converts a PyTorch tensor (CPU or GPU) or other array-like data to a NumPy array.
    """
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data.cpu().numpy()
        return data.numpy()
    return np.asarray(data)

# --- Quaternion and Transformation Helper Functions ---


def quat_to_rotation_matrix(q_wxyz):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    # Ensure q_wxyz is a 1D array if it's a single quaternion
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
    elif q_wxyz.ndim == 2:  # Batch of quaternions (num_envs, 4)
        # This function is generally used for single quaternions in the current logic.
        # If batched processing of rotation matrices is needed elsewhere, this would need vectorization.
        raise NotImplementedError(
            "Batched quaternion to rotation matrix not directly implemented here. Process one by one if needed.")
    else:
        raise ValueError(f"Unexpected quaternion shape: {q_wxyz.shape}")
    return R


def transform_point_to_local_frame(world_point, box_world_pos, box_world_orient_quat_wxyz):
    """
    Transforms a point from world coordinates to a box's local coordinate system.
    Handles single point or a batch of points if world_point is (B,3) and box params are for one box.
    If box_world_pos and box_world_orient_quat_wxyz are also batched (for different boxes),
    this function would need further adaptation. Currently assumes they are for a single box context.
    """
    single_op = world_point.ndim == 1

    if single_op:  # Convert single point to a batch of 1 for consistent processing
        world_point_b = world_point.reshape(1, -1)
        box_world_pos_b = box_world_pos.reshape(1, -1)
        box_world_orient_quat_wxyz_b = box_world_orient_quat_wxyz.reshape(
            1, -1)
    else:  # Already batched points, but box params are singular for this call context
        world_point_b = world_point
        # Repeat box params to match batch size of points
        num_points = world_point.shape[0]
        box_world_pos_b = np.tile(box_world_pos, (num_points, 1))
        box_world_orient_quat_wxyz_b = np.tile(
            box_world_orient_quat_wxyz, (num_points, 1))

    vec_world_b = world_point_b - box_world_pos_b  # (B, 3)

    local_points_list = []
    for i in range(world_point_b.shape[0]):  # Iterate over the batch of points
        # quat_to_rotation_matrix expects a single (4,) quaternion
        R_world_from_local_i = quat_to_rotation_matrix(
            box_world_orient_quat_wxyz_b[i])
        R_local_from_world_i = R_world_from_local_i.T
        local_points_list.append(R_local_from_world_i @ vec_world_b[i])

    local_points_arr = np.array(local_points_list, dtype=np.float32)
    return local_points_arr[0] if single_op else local_points_arr


# --- Voxelization Helper Function ---
def boxes_to_voxel_grid_batched(list_of_batched_box_params, grid_dims, grid_origins_batch, voxel_size, num_envs):
    """
    Voxelizes lists of oriented boxes for a batch of environments.
    Returns a batched voxel grid: (num_envs, D, H, W).
    `list_of_batched_box_params` is a list (per component type) of dicts,
    where 'position' and 'orientation_quat' are batched (num_envs, ...), and 'size' is fixed.
    """
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
    """
    Voxelizes a list of oriented boxes into a 3D grid for a single environment.
    """
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


class FrankaShelfEnv(gym.Env):
    """
    Franka environment with reconfigurable shelves, supporting batched operations
    for Genesis native parallel environments.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

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

    def __init__(self, render_mode=None, num_envs=1, env_spacing=(2.0, 2.0),
                 voxel_grid_dims=(16, 16, 16), workspace_bounds=((-0.8, 0.8), (-0.8, 0.8), (0.0, 1.5)),
                 voxel_grid_world_size=(1.6, 1.6, 1.0), max_steps_per_episode=500, dt=0.01,
                 franka_xml_path='xml/franka_emika_panda/panda.xml',
                 k_dist_reward=1.0, k_time_penalty=0.01, k_action_penalty=0.001,
                 k_joint_limit_penalty=10.0, k_collision_penalty=100.0,
                 success_reward_val=200.0, success_threshold_val=0.05,
                 k_accel_penalty=0.0):  # Added k_accel_penalty with a default of 0.0
        super().__init__()

        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.render_mode = render_mode
        self.voxel_grid_dims = voxel_grid_dims
        self.workspace_bounds_x, self.workspace_bounds_y, self.workspace_bounds_z = workspace_bounds
        self.voxel_grid_world_span_x, self.voxel_grid_world_span_y, self.voxel_grid_world_span_z = voxel_grid_world_size

        _vx, _vy, _vz = self.voxel_grid_world_span_x / \
            self.voxel_grid_dims[2], self.voxel_grid_world_span_y / \
            self.voxel_grid_dims[1], self.voxel_grid_world_span_z / \
            self.voxel_grid_dims[0]
        self.voxel_size = min(_vx, _vy, _vz)
        eff_w, eff_h, eff_d = self.voxel_grid_dims[2]*self.voxel_size, self.voxel_grid_dims[1] * \
            self.voxel_size, self.voxel_grid_dims[0]*self.voxel_size

        self.grid_origin_x = self.workspace_bounds_x[0] + (
            (self.workspace_bounds_x[1]-self.workspace_bounds_x[0]) - eff_w)/2.0
        self.grid_origin_y = self.workspace_bounds_y[0] + (
            (self.workspace_bounds_y[1]-self.workspace_bounds_y[0]) - eff_h)/2.0
        self.grid_origin_z = self.workspace_bounds_z[0]
        self.grid_origin_global = np.array(
            [self.grid_origin_x, self.grid_origin_y, self.grid_origin_z], dtype=np.float32)
        self.grid_origins_batch = np.tile(
            self.grid_origin_global, (self.num_envs, 1))

        # Adjust grid origins based on env_spacing for batched environments
        # This assumes robots are effectively at (i*env_spacing_x, j*env_spacing_y, 0) in world
        # and the voxel grid should be relative to each robot's local origin.
        # For simplicity, if env_spacing is used, we assume the voxel grid moves with the env.
        # If scene.build() offsets the *entire* sub-scene including the robot from (0,0,0),
        # then target positions and voxel grids need to be in that sub-scene's frame or transformed.
        # The current logic places shelves relative to a global (0,0,0) for the robot base.
        # If env_spacing moves the robot bases, then shelf positions also need to be offset.
        # Let's assume for now that `env_spacing` in `scene.build` means each env's (0,0,0)
        # is effectively shifted in the global Genesis coordinate system.
        # If so, all world coordinates (shelf pos, target pos) need to be generated per-env
        # relative to that env's shifted origin.
        # This part requires careful understanding of Genesis's `env_spacing` behavior.
        # For now, the `grid_origins_batch` is a simple tile, implying all voxel grids share the same world origin.
        # This might be incorrect if `env_spacing` shifts the entire coordinate system for each env.

        self.max_steps = max_steps_per_episode
        self.current_step = np.zeros(self.num_envs, dtype=np.int32)
        self.dt = dt
        self.franka_xml_path = franka_xml_path

        self.k_dist_reward, self.k_time_penalty, self.k_action_penalty = k_dist_reward, k_time_penalty, k_action_penalty
        self.k_joint_limit_penalty, self.k_collision_penalty = k_joint_limit_penalty, k_collision_penalty
        self.success_reward, self.success_threshold = success_reward_val, success_threshold_val
        self.k_accel_penalty = k_accel_penalty  # Storing the new penalty coefficient

        self._define_shelf_configurations()
        self.current_shelf_config_key = [""] * self.num_envs
        self.current_shelf_instance_params = [{} for _ in range(self.num_envs)]

        viewer_options = gs.options.ViewerOptions(camera_pos=(1.8, -1.8, 2.0), camera_lookat=(
            0.3, 0.0, 0.5), camera_fov=45, res=(960, 640), max_FPS=self.metadata['render_fps'])
        # Show viewer if human mode, regardless of num_envs, though Genesis might only show one.
        self.scene = gs.Scene(viewer_options=viewer_options, sim_options=gs.options.SimOptions(
            dt=self.dt), show_viewer=(self.render_mode == "human"))

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=self.franka_xml_path))
        self.shelf_component_entities = [self.scene.add_entity(gs.morphs.Box(pos=(0, -10-i*0.5, 0), quat=(1, 0, 0, 0), size=tuple(
            s), fixed=True, visualization=True, collision=True)) for i, s in enumerate(self.FIXED_COMPONENT_SIZES)]

        if not self.scene.is_built:
            print(
                f"Building Genesis scene for {self.num_envs} environments with spacing {self.env_spacing}...")
            self.scene.build(n_envs=self.num_envs,
                             env_spacing=self.env_spacing)
            print("Scene built.")
        else:
            print("Warning: Scene was already built before multi-env setup.")

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

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.FRANKA_NUM_ARM_JOINTS,), dtype=np.float32)
        self.max_joint_velocity_scale = 0.5

        robot_state_flat_dim = self.FRANKA_NUM_ARM_JOINTS*2+3+4
        relative_target_pos_dim = 3
        self.observation_space = spaces.Dict({
            "robot_state_flat": spaces.Box(low=np.full(robot_state_flat_dim, -np.inf, dtype=np.float32), high=np.full(robot_state_flat_dim, np.inf, dtype=np.float32), dtype=np.float32),
            "relative_target_pos": spaces.Box(low=np.full(relative_target_pos_dim, -np.inf, dtype=np.float32), high=np.full(relative_target_pos_dim, np.inf, dtype=np.float32), dtype=np.float32),
            "obstacle_voxels": spaces.Box(low=0, high=1, shape=self.voxel_grid_dims, dtype=np.float32)
        })

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

        if self.render_mode == "human":  # User reverted condition to always draw if human mode
            if hasattr(self.scene, 'clear_debug_objects'):
                self.scene.clear_debug_objects()
            # Pass all target positions if draw_debug_spheres supports it.
            # Convert (B,3) numpy array to list of lists/tuples if required by Genesis API.
            # For example: target_list_for_genesis = [list(pos) for pos in self.target_position_world]
            # Assuming draw_debug_spheres can take a (B,3) numpy array or list of arrays:
            self.scene.draw_debug_spheres(
                self.target_position_world.tolist(), radius=0.03, color=(0, 1, 0, 0.8))
        return self.target_position_world

    def _get_obs_batched(self):
        arm_joint_pos, arm_joint_vel, ee_pos, ee_orient_quat_wxyz = self._get_robot_state_parts_batched()
        robot_state_flat = np.concatenate(
            [arm_joint_pos, arm_joint_vel, ee_pos, ee_orient_quat_wxyz], axis=1).astype(np.float32)
        relative_target_pos = (
            self.target_position_world - ee_pos).astype(np.float32)
        return {"robot_state_flat": robot_state_flat, "relative_target_pos": relative_target_pos, "obstacle_voxels": self.current_voxel_grid.copy()}

    def reset(self, seed=None, options=None):
        # Adhering to Gymnasium's reset signature
        super().reset(seed=seed)  # Call to gym.Env.reset() for seeding internal PRNG
        if seed is not None:  # Seed Python's random and NumPy if a seed is provided
            np.random.seed(seed)
            random.seed(seed)
            # Note: Genesis specific global seeding might also be needed if it has its own RNG.
            # For now, assuming gs.Scene.reset() and entity states cover sim determinism.

        if self.scene.is_built:
            self.scene.reset()
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
        # Initialize with empty dicts
        infos_batch = [{} for _ in range(self.num_envs)]
        # Populate shelf_config in info as it's set during reset
        for i in range(self.num_envs):
            infos_batch[i]["shelf_config"] = self.current_shelf_config_key[i]

        if self.num_envs == 1:
            single_obs = {key: val[0] for key, val in obs_batch.items()}
            return single_obs, infos_batch[0]  # Return obs, info
        else:
            # For batched environments not wrapped by SB3 VecEnv (e.g. internal testing)
            return obs_batch, infos_batch

    def step(self, actions_batch):
        if actions_batch.ndim == 1 and self.num_envs == 1:
            actions_batch = actions_batch.reshape(1, -1)
        assert actions_batch.shape == (
            self.num_envs, self.FRANKA_NUM_ARM_JOINTS), f"Expected actions_batch shape {(self.num_envs, self.FRANKA_NUM_ARM_JOINTS)}, got {actions_batch.shape}"
        self.current_step += 1
        actions_batch_clipped = np.clip(
            actions_batch, -1.0, 1.0).astype(np.float32)
        scaled_actions_batch = actions_batch_clipped * self.max_joint_velocity_scale
        actions_tensor = torch.tensor(
            scaled_actions_batch, device=gs.device, dtype=torch.float32)
        self.franka.control_dofs_velocity(
            actions_tensor, self.franka_arm_dof_indices)
        self.scene.step()
        arm_joint_pos_b, arm_joint_vel_b, ee_pos_b, _ = self._get_robot_state_parts_batched()
        dist_to_target_b = np.linalg.norm(
            ee_pos_b - self.target_position_world, axis=1)
        reward_distance_b = -self.k_dist_reward * dist_to_target_b
        time_penalty_b = -self.k_time_penalty * \
            np.ones(self.num_envs, dtype=np.float32)
        action_penalty_b = -self.k_action_penalty * \
            np.sum(np.square(actions_batch_clipped),
                   axis=1)  # Penalize clipped actions
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

        # --- Collision Penalty (Modified Part) ---
        collision_penalty_val_b = np.zeros(self.num_envs, dtype=np.float32)
        # Track collision termination per env
        terminated_collision_b = np.zeros(self.num_envs, dtype=bool)

        if self.scene.is_built and hasattr(self.scene, 'get_contacts'):
            contacts = self.scene.get_contacts()  # This should return contacts for all envs
            if contacts:  # Check if contacts list is not empty
                franka_entity_id = self.franka.entity_id
                # Create a set of shelf entity IDs for efficient lookup
                shelf_entity_ids = {
                    comp.entity_id for comp in self.shelf_component_entities}

                for contact_info in contacts:
                    # Assuming contact_info is an object with attributes: env_idx, entity0_id, entity1_id
                    env_idx = contact_info.env_idx
                    if env_idx >= self.num_envs:  # Safety check for env_idx
                        continue

                    e0_id = contact_info.entity0_id
                    e1_id = contact_info.entity1_id

                    is_franka0 = (e0_id == franka_entity_id)
                    is_franka1 = (e1_id == franka_entity_id)
                    is_shelf0 = (e0_id in shelf_entity_ids)
                    is_shelf1 = (e1_id in shelf_entity_ids)

                    # Check for Franka self-collision (franka vs franka)
                    # or Franka-shelf collision (franka vs shelf component)
                    # Self-collision (ensure not same link if detailed)
                    if (is_franka0 and is_franka1 and e0_id != e1_id):
                        collision_penalty_val_b[env_idx] -= self.k_collision_penalty
                        terminated_collision_b[env_idx] = True
                    elif (is_franka0 and is_shelf1) or (is_franka1 and is_shelf0):  # Shelf collision
                        collision_penalty_val_b[env_idx] -= self.k_collision_penalty
                        terminated_collision_b[env_idx] = True
        # --- End of Collision Penalty Modification ---

        success_reward_val_b = np.zeros(self.num_envs, dtype=np.float32)
        terminated_success_b = dist_to_target_b < self.success_threshold
        success_reward_val_b[terminated_success_b] = self.success_reward

        # --- Smoothness Penalty (Acceleration) (New Part) ---
        accel_penalty_b = np.zeros(self.num_envs, dtype=np.float32)
        if self.k_accel_penalty > 0 and self.dt > 0:
            joint_accel_b = (arm_joint_vel_b - self.prev_joint_vel) / self.dt
            accel_penalty_b = -self.k_accel_penalty * \
                np.sum(np.square(joint_accel_b), axis=1)
        # --- End of Smoothness Penalty ---

        rewards_b = reward_distance_b + time_penalty_b + action_penalty_b + \
            joint_pos_penalty_b + collision_penalty_val_b + success_reward_val_b + \
            accel_penalty_b  # Added accel_penalty_b

        terminated_b = np.logical_or(
            terminated_collision_b, terminated_success_b)  # terminated_collision_b is now properly calculated
        truncated_b = self.current_step >= self.max_steps
        observations_b = self._get_obs_batched()
        infos_b = []
        for i in range(self.num_envs):
            infos_b.append({"is_success": terminated_success_b[i],
                            "distance_to_target": float(dist_to_target_b[i]),
                            # This now reflects actual collisions
                            "collision_detected": terminated_collision_b[i],
                            "current_step": self.current_step[i],
                            "shelf_config": self.current_shelf_config_key[i]})
        self.prev_joint_vel = arm_joint_vel_b.copy()
        if self.num_envs == 1:
            # Gymnasium expects obs, reward, terminated, truncated, info
            return observations_b[0], float(rewards_b[0]), terminated_b[0], truncated_b[0], infos_b[0]
        else:
            # For batched envs not wrapped by SB3 VecEnv
            return observations_b, rewards_b, terminated_b, truncated_b, infos_b

    def render(self, mode='human'):
        if self.render_mode == "human":
            return None
        elif self.render_mode == "rgb_array":
            if self.num_envs == 1 and hasattr(self, 'current_voxel_grid'):
                grid_to_render = self.current_voxel_grid[0]
                img_slice = np.sum(grid_to_render, axis=0)
                img_slice_norm = np.clip(img_slice, 0, np.max(
                    img_slice) if np.max(img_slice) > 0 else 1.0)
                if np.max(img_slice_norm) > 0:
                    img_slice_norm = img_slice_norm / np.max(img_slice_norm)
                img_slice_uint8 = (img_slice_norm * 255).astype(np.uint8)
                return np.stack([img_slice_uint8]*3, axis=-1)
            return np.zeros((self.num_envs, self.voxel_grid_dims[1], self.voxel_grid_dims[2], 3), dtype=np.uint8) if self.num_envs > 1 else np.zeros((self.voxel_grid_dims[1], self.voxel_grid_dims[2], 3), dtype=np.uint8)
        return None

    def close(self):
        if hasattr(self, 'scene') and self.scene is not None:
            if hasattr(self.scene, 'close'):
                self.scene.close()
            elif hasattr(self.scene, 'shutdown'):
                self.scene.shutdown()
        print("FrankaShelfEnv closed.")


if __name__ == '__main__':
    print("Attempting to initialize Genesis...")
    try:
        gs.init(backend=gs.cpu)
        print("Genesis initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Genesis: {e}\nExiting.")
        exit()

    NUM_PARALLEL_ENVS_TO_TEST = 4
    print(
        f"Attempting to create FrankaShelfEnv with num_envs={NUM_PARALLEL_ENVS_TO_TEST}...")
    env = None
    try:
        render_mode_for_test = "human"  # User reverted to always try human rendering
        env = FrankaShelfEnv(num_envs=NUM_PARALLEL_ENVS_TO_TEST,
                             render_mode=render_mode_for_test,
                             k_collision_penalty=150.0,  # Example: Test with a specific collision penalty
                             k_accel_penalty=0.001      # Example: Test with a small accel penalty
                             )
        print("FrankaShelfEnv created successfully.")
        num_episodes_to_run = 2
        print(f"Running {num_episodes_to_run} example episodes...")

        for episode_num in range(num_episodes_to_run):
            print(
                f"\n--- Starting Episode {episode_num + 1} (Batch of {env.num_envs} envs) ---")
            obs_batch, infos_batch = env.reset()
            if env.num_envs == 1:  # Should not happen with NUM_PARALLEL_ENVS_TO_TEST = 4 but good for robustness
                obs_batch = {key: val[np.newaxis, ...]
                             for key, val in obs_batch.items()}
                # Ensure infos_batch is a list of dicts
                infos_batch = [infos_batch]
            print(f"  Reset successful.")
            for i in range(env.num_envs):
                current_info = infos_batch[i] if isinstance(
                    infos_batch, list) and i < len(infos_batch) else {}
                print(
                    f"    Env {i}: Shelf Config: {current_info.get('shelf_config', env.current_shelf_config_key[i])}")
                print(
                    f"    Env {i}: Initial Relative Target (Obs): {obs_batch['relative_target_pos'][i]}")
                print(
                    f"    Env {i}: Shelf Yaw (deg): {env.current_shelf_instance_params[i].get('shelf_assembly_yaw', 0) * 180/np.pi:.1f}")
                print(
                    f"    Env {i}: Voxel sum: {np.sum(obs_batch['obstacle_voxels'][i])}")

            max_steps_this_episode = 50
            for step_num in range(max_steps_this_episode):
                actions_batch = np.array(
                    [env.action_space.sample() for _ in range(env.num_envs)])
                obs_batch, rewards_batch, terminated_batch, truncated_batch, infos_batch = env.step(
                    actions_batch)
                if env.num_envs == 1:  # Should not happen with NUM_PARALLEL_ENVS_TO_TEST = 4
                    obs_batch = {key: val[np.newaxis, ...]
                                 for key, val in obs_batch.items()}
                    rewards_batch = np.array([rewards_batch])
                    terminated_batch = np.array([terminated_batch])
                    truncated_batch = np.array([truncated_batch])
                    # Ensure infos_batch is a list of dicts
                    infos_batch = [infos_batch]

                if (step_num + 1) % 10 == 0:
                    print(f"  Step {step_num+1}:")
                    for i in range(env.num_envs):
                        current_info = infos_batch[i] if isinstance(
                            infos_batch, list) and i < len(infos_batch) else {}
                        print(
                            f"    Env {i}: Dist={current_info.get('distance_to_target', -1):.2f}, Rew={rewards_batch[i]:.2f}, Term={terminated_batch[i]}, Trunc={truncated_batch[i]}, Coll={current_info.get('collision_detected', False)}")
                if np.all(np.logical_or(terminated_batch, truncated_batch)):
                    print(f"  All environments finished at step {step_num+1}.")
                    break
            if env.render_mode == "human":  # User reverted condition
                print("  Pausing for 1s...")
                import time
                time.sleep(1)
    except Exception as e:
        import traceback
        print(f"ERROR during FrankaShelfEnv example usage: {e}")
        traceback.print_exc()
    finally:
        if env:
            print("\nClosing FrankaShelfEnv...")
            env.close()
        if hasattr(gs, 'shutdown'):
            print("Shutting down Genesis...")
            gs.shutdown()
        print("Example application finished.")
