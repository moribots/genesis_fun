import gymnasium as gym
from gymnasium import spaces
import numpy as np
# from scipy.spatial import ConvexHull # No longer needed
import torch  # For type checking and tensor operations if Genesis returns PyTorch tensors
import random

# Import Genesis
import genesis as gs  # Assuming 'genesis' is the correct import name

# Initialize Genesis Simulator
# Assuming GPU backend, call this once globally if needed
gs.init(backend=gs.gpu)


def to_numpy(data):
    """
    Converts a PyTorch tensor (CPU or GPU) or other array-like data to a NumPy array.
    """
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data.cpu().numpy()
        return data.numpy()
    return np.asarray(data)


# --- Helper Functions ---

def boxes_to_voxel_grid(list_of_box_params, grid_dims, grid_origin, voxel_size):
    """
    Voxelizes a list of boxes into a 3D grid.

    A voxel is marked as occupied (1.0) if its center point lies inside any of the
    provided boxes.

    Args:
        list_of_box_params (list of dict): A list where each element is a dictionary
            containing 'position' (center_x, center_y, center_z) and
            'size' (size_x, size_y, size_z) for a box.
        grid_dims (tuple): A tuple (D, H, W) specifying the dimensions (Depth, Height, Width)
            of the voxel grid.
        grid_origin (tuple): A tuple (x, y, z) representing the world coordinates of the
            origin (corner) of the voxel grid.
        voxel_size (float): The side length of a single cubic voxel. Assumes uniform voxel size.

    Returns:
        numpy.ndarray: A 3D numpy array of shape `grid_dims` representing the voxel grid,
            with 1.0 for occupied voxels and 0.0 for empty voxels.
    """
    grid = np.zeros(grid_dims, dtype=np.float32)
    grid_D, grid_H, grid_W = grid_dims
    origin_x, origin_y, origin_z = grid_origin

    if not list_of_box_params:
        return grid

    # Iterate through each voxel in the grid
    for k in range(grid_D):  # Depth dimension
        for j in range(grid_H):  # Height dimension
            for i in range(grid_W):  # Width dimension
                # Calculate the center coordinates of the current voxel
                voxel_center_x = origin_x + (i + 0.5) * voxel_size
                voxel_center_y = origin_y + (j + 0.5) * voxel_size
                voxel_center_z = origin_z + (k + 0.5) * voxel_size

                # Check if this voxel center is inside any of the boxes
                for box_params in list_of_box_params:
                    box_pos = box_params['position']
                    box_size = box_params['size']

                    half_size_x = box_size[0] / 2.0
                    half_size_y = box_size[1] / 2.0
                    half_size_z = box_size[2] / 2.0

                    min_x, max_x = box_pos[0] - \
                        half_size_x, box_pos[0] + half_size_x
                    min_y, max_y = box_pos[1] - \
                        half_size_y, box_pos[1] + half_size_y
                    min_z, max_z = box_pos[2] - \
                        half_size_z, box_pos[2] + half_size_z

                    if (min_x <= voxel_center_x <= max_x and
                        min_y <= voxel_center_y <= max_y and
                            min_z <= voxel_center_z <= max_z):
                        grid[k, j, i] = 1.0  # Mark voxel as occupied
                        break  # Move to the next voxel once occupied by one box
    return grid


class FrankaShelfEnv(gym.Env):
    """
    A Gymnasium environment for a Franka Emika Panda robot interacting with
    randomly generated shelf obstacles (represented as boxes). The robot's goal is to reach arbitrary
    keypoints while avoiding collisions. Observations include robot state,
    target position, and a voxelized representation of obstacles.

    The environment uses the Genesis simulator for physics and rendering.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    FRANKA_JOINT_NAMES = [  # 7 arm joints + 2 finger joints
        'joint1', 'joint2', 'joint3', 'joint4',
        'joint5', 'joint6', 'joint7',
        'finger_joint1', 'finger_joint2',
    ]
    FRANKA_NUM_ARM_JOINTS = 7
    FRANKA_NUM_TOTAL_JOINTS = len(FRANKA_JOINT_NAMES)
    # Default initial position for the Franka arm (can be customized)
    FRANKA_DEFAULT_INITIAL_QPOS = np.array(
        [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])

    def __init__(self, render_mode=None,
                 voxel_grid_dims=(16, 16, 16),  # D, H, W
                 # X, Y, Z for robot workspace & target
                 workspace_bounds=((-0.7, 0.7), (-0.7, 0.7), (0.0, 1.4)),
                 # Physical size of the voxelized area (m)
                 voxel_grid_world_size=(1.4, 1.4, 1.4),
                 max_shelves=3,
                 # Min (x,y,z) size for a shelf box
                 min_shelf_size=(0.2, 0.2, 0.05),
                 # Max (x,y,z) size for a shelf box
                 max_shelf_size=(0.6, 0.4, 0.3),
                 max_steps_per_episode=500,
                 dt=0.01,  # Simulation timestep
                 franka_xml_path='xml/franka_emika_panda/panda.xml'  # Path to Franka MJCF
                 ):
        """
        Initializes the Franka Shelf Environment.
        """
        super().__init__()

        self.render_mode = render_mode
        self.voxel_grid_dims = voxel_grid_dims
        self.workspace_bounds_x, self.workspace_bounds_y, self.workspace_bounds_z = workspace_bounds
        self.voxel_grid_world_size_x, self.voxel_grid_world_size_y, self.voxel_grid_world_size_z = voxel_grid_world_size

        self.voxel_size_x = self.voxel_grid_world_size_x / \
            self.voxel_grid_dims[2]  # W
        self.voxel_size_y = self.voxel_grid_world_size_y / \
            self.voxel_grid_dims[1]  # H
        self.voxel_size_z = self.voxel_grid_world_size_z / \
            self.voxel_grid_dims[0]  # D
        self.voxel_size = min(
            self.voxel_size_x, self.voxel_size_y, self.voxel_size_z)

        self.grid_origin_x = self.workspace_bounds_x[0] + (
            self.workspace_bounds_x[1] - self.workspace_bounds_x[0] - self.voxel_grid_world_size_x) / 2
        self.grid_origin_y = self.workspace_bounds_y[0] + (
            self.workspace_bounds_y[1] - self.workspace_bounds_y[0] - self.voxel_grid_world_size_y) / 2
        self.grid_origin_z = self.workspace_bounds_z[0]
        self.grid_origin = (self.grid_origin_x,
                            self.grid_origin_y, self.grid_origin_z)

        self.max_shelves = max_shelves
        self.min_shelf_size = np.array(min_shelf_size)
        self.max_shelf_size = np.array(max_shelf_size)
        self.max_steps = max_steps_per_episode
        self.current_step = 0
        self.dt = dt
        self.franka_xml_path = franka_xml_path

        viewer_options = gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.8),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
            res=(960, 640),
            max_FPS=self.metadata['render_fps'],
        )

        self.scene = gs.Scene(
            viewer_options=viewer_options,
            sim_options=gs.options.SimOptions(dt=self.dt),
            show_viewer=(self.render_mode == "human")
        )

        self.plane = self.scene.add_entity(
            gs.morphs.Plane())

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file=self.franka_xml_path))

        self.target_position_world = np.zeros(3, dtype=np.float32)
        self.target_visualization_entity = None

        self.num_shelves_to_gen = random.randint(1, self.max_shelves)
        self.shelf_entities = []
        self.shelf_sizes = []
        for i in range(self.num_shelves_to_gen):
            # Randomly determine box size
            shelf_size_x = random.uniform(
                self.min_shelf_size[0], self.max_shelf_size[0])
            shelf_size_y = random.uniform(
                self.min_shelf_size[1], self.max_shelf_size[1])
            shelf_size_z = random.uniform(
                self.min_shelf_size[2], self.max_shelf_size[2])
            self.shelf_sizes.append((shelf_size_x, shelf_size_y, shelf_size_z))

        box_params_for_voxelization = self._generate_random_shelves()
        self.target_position_world = self._generate_random_target()

        self.current_voxel_grid = boxes_to_voxel_grid(  # Updated function call
            box_params_for_voxelization,
            self.voxel_grid_dims,
            self.grid_origin,
            self.voxel_size
        )

        self.franka_dof_indices = np.array([self.franka.get_joint(
            name).dof_idx_local for name in self.FRANKA_JOINT_NAMES])
        self.franka_arm_dof_indices = self.franka_dof_indices[:self.FRANKA_NUM_ARM_JOINTS]
        self.robot_ee_link_name = "hand"

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.FRANKA_NUM_ARM_JOINTS,), dtype=np.float32)
        self.max_joint_velocity_scale = 0.5

        robot_state_dim = self.FRANKA_NUM_ARM_JOINTS * 2 + 3 + 4
        target_pos_dim = 3

        self.observation_space = spaces.Dict({
            "robot_state": spaces.Box(low=-np.inf, high=np.inf, shape=(robot_state_dim,), dtype=np.float32),
            "target_position": spaces.Box(
                low=np.array([self.workspace_bounds_x[0], self.workspace_bounds_y[0],
                             self.workspace_bounds_z[0]], dtype=np.float32),
                high=np.array([self.workspace_bounds_x[1], self.workspace_bounds_y[1],
                              self.workspace_bounds_z[1]], dtype=np.float32),
                shape=(target_pos_dim,), dtype=np.float32
            ),
            "obstacle_voxels": spaces.Box(low=0, high=1, shape=self.voxel_grid_dims, dtype=np.float32)
        })

    def _set_franka_gains(self):
        """Helper to set default PD gains for Franka if needed."""
        kp_arm = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000])
        kp_fingers = np.array([100, 100])
        kv_arm = np.array([450, 450, 350, 350, 200, 200, 200])
        kv_fingers = np.array([10, 10])

        self.franka.set_dofs_kp(np.concatenate(
            [kp_arm, kp_fingers]), self.franka_dof_indices)
        self.franka.set_dofs_kv(np.concatenate(
            [kv_arm, kv_fingers]), self.franka_dof_indices)

    def _get_robot_state(self):
        """Retrieves the current state of the Franka robot."""
        joint_pos_all = to_numpy(
            self.franka.get_dofs_position(self.franka_dof_indices))
        joint_vel_all = to_numpy(
            self.franka.get_dofs_velocity(self.franka_dof_indices))

        arm_joint_pos = joint_pos_all[:self.FRANKA_NUM_ARM_JOINTS]
        arm_joint_vel = joint_vel_all[:self.FRANKA_NUM_ARM_JOINTS]

        ee_link = self.franka.get_link(self.robot_ee_link_name)
        ee_pos = to_numpy(ee_link.get_pos())
        ee_orient_quat = to_numpy(ee_link.get_quat())

        return np.concatenate([
            arm_joint_pos, arm_joint_vel, ee_pos, ee_orient_quat
        ]).astype(np.float32)

    def _generate_random_shelves(self):
        """
        Generates random box shelves and adds them to the Genesis scene.
        Returns a list of box parameters (position, size) for voxelization.
        """

        list_of_box_params_for_voxelization = []

        for i in range(self.num_shelves_to_gen):
            current_shelf_size = self.shelf_sizes[i]

            # Randomly determine box position (center of the box)
            # Ensure the box is within workspace and above ground
            pos_x = random.uniform(
                # Add some margin from workspace edge
                self.workspace_bounds_x[0] + current_shelf_size[0] / 2 + 0.1,
                self.workspace_bounds_x[1] - current_shelf_size[0] / 2 - 0.1)
            pos_y = random.uniform(
                self.workspace_bounds_y[0] + current_shelf_size[1] / 2 + 0.1,
                self.workspace_bounds_y[1] - current_shelf_size[1] / 2 - 0.1)
            pos_z = random.uniform(  # Place base of box on or above ground
                # Small offset from ground
                self.workspace_bounds_z[0] + current_shelf_size[2] / 2 + 0.05,
                self.workspace_bounds_z[1] * 0.7 - current_shelf_size[2] / 2)  # Keep it in lower part
            current_shelf_pos = (pos_x, pos_y, pos_z)

            if not self.scene.is_built:
                shelf_entity = self.scene.add_entity(
                    gs.morphs.Box(
                        pos=current_shelf_pos,
                        size=current_shelf_size,
                        fixed=True,  # Same as is_static=True
                        visualization=True,  # Ensure it's visible
                        collision=True,  # Ensure it's collidable
                    )
                )
                self.shelf_entities.append(shelf_entity)
            else:
                self.shelf_entities[i].set_pos(current_shelf_pos)
            list_of_box_params_for_voxelization.append(
                {'position': np.array(current_shelf_pos),
                    'size': np.array(current_shelf_size)}
            )

        if not self.scene.is_built:
            self.scene.build()
        return list_of_box_params_for_voxelization

    def _generate_random_target(self):
        """Generates a random target position within the workspace and visualizes it."""
        x = random.uniform(
            self.workspace_bounds_x[0] + 0.1, self.workspace_bounds_x[1] - 0.1)
        y = random.uniform(
            self.workspace_bounds_y[0] + 0.1, self.workspace_bounds_y[1] - 0.1)
        z = random.uniform(
            self.workspace_bounds_z[0] + 0.2, self.workspace_bounds_z[1] - 0.2)
        self.target_position_world = np.array([x, y, z], dtype=np.float32)

        if self.target_visualization_entity is not None:
            self.scene.clear_debug_objects()
            self.target_visualization_entity = None

        self.target_visualization_entity = self.scene.draw_debug_spheres(
            [self.target_position_world],
            radius=0.03,
            color=(0, 1, 0, 0.8)
        )
        return self.target_position_world

    def _get_obs(self):
        """Constructs the observation dictionary."""
        robot_state = self._get_robot_state()
        return {
            "robot_state": robot_state,
            "target_position": self.target_position_world.copy(),
            "obstacle_voxels": self.current_voxel_grid.copy()
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.scene.reset()
        self.current_step = 0

        self.franka.set_dofs_position(
            self.FRANKA_DEFAULT_INITIAL_QPOS, self.franka_dof_indices)
        self.franka.set_dofs_velocity(
            np.zeros(self.FRANKA_NUM_TOTAL_JOINTS), self.franka_dof_indices)

        for _ in range(5):
            self.scene.step()

        box_params_for_voxelization = self._generate_random_shelves()
        self.target_position_world = self._generate_random_target()

        self.current_voxel_grid = boxes_to_voxel_grid(  # Updated function call
            box_params_for_voxelization,
            self.voxel_grid_dims,
            self.grid_origin,
            self.voxel_size
        )

        observation = self._get_obs()
        info = {"initial_target": self.target_position_world.copy()}

        return observation, info

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1
        scaled_action_arm_velocities = action * self.max_joint_velocity_scale
        self.franka.control_dofs_velocity(
            scaled_action_arm_velocities, self.franka_arm_dof_indices)
        self.scene.step()

        current_robot_state = self._get_robot_state()
        current_ee_pos = current_robot_state[self.FRANKA_NUM_ARM_JOINTS *
                                             2: self.FRANKA_NUM_ARM_JOINTS*2+3]

        dist_to_target = np.linalg.norm(
            current_ee_pos - self.target_position_world)
        reward_distance = -dist_to_target

        # 2. Collision penalty
        collision_penalty = 0

        # contacts = self.scene.get_contacts() # Genesis API for contacts
        # for contact_pair in contacts:
        # entity_a = contact_pair.entity_a
        # entity_b = contact_pair.entity_b
        # Check if franka is involved and the other entity is a shelf or franka itself (self-collision)
        # is_franka_involved = (entity_a == self.franka.id or entity_b == self.franka.id)
        # is_shelf_involved = False
        # for shelf_entity in self.shelf_entities:
        # if entity_a == shelf_entity.id or entity_b == shelf_entity.id:
        # is_shelf_involved = True
        # break
        # This logic needs refinement based on how Genesis reports contacts and entity IDs
        # if is_franka_involved and (is_shelf_involved or (entity_a == self.franka.id and entity_b == self.franka.id)):
        # Differentiate self-collision vs environment collision if needed
        # collision_penalty = -100 # Large negative reward for any collision
        # break
        # Placeholder collision:
        if random.random() < 0.005:  # Simulate a very small chance of collision for testing
            collision_penalty = -100

        success_reward = 0
        terminated_success = False
        success_threshold = 0.05
        if dist_to_target < success_threshold:
            success_reward = 200
            terminated_success = True

        action_magnitude_penalty = -0.01 * np.sum(np.square(action))
        reward = reward_distance + collision_penalty + \
            success_reward + action_magnitude_penalty

        terminated_collision = collision_penalty < 0
        terminated = terminated_collision or terminated_success
        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()
        info = {
            "is_success": terminated_success,
            "distance_to_target": float(dist_to_target),
            "collision_detected": terminated_collision
        }

        return observation, float(reward), terminated, truncated, info

    def render(self):
        """Renders the environment."""
        if self.render_mode == "human":
            return None
        elif self.render_mode == "rgb_array":
            if hasattr(self, 'current_voxel_grid'):
                img_slice = np.sum(self.current_voxel_grid, axis=0)
                img_slice = np.clip(img_slice, 0, 1) * 255
                return np.stack([img_slice]*3, axis=-1).astype(np.uint8)
            return np.zeros((self.voxel_grid_dims[1], self.voxel_grid_dims[2], 3), dtype=np.uint8)

    def close(self):
        """Performs any necessary cleanup."""
        print("FrankaShelfEnv closed.")


if __name__ == '__main__':
    print("Attempting to create FrankaShelfEnv with Genesis...")
    env = None
    try:
        env = FrankaShelfEnv(
            render_mode="human",
        )
        print("FrankaShelfEnv created.")
        obs, info = env.reset()
        print("Reset successful.")
        print("Initial observation (robot_state shape):",
              obs["robot_state"].shape)
        print("Initial observation (target_position):", obs["target_position"])
        print("Initial observation (obstacle_voxels shape):",
              obs["obstacle_voxels"].shape)
        print(
            f"Voxel grid sum (occupied voxels in reset): {np.sum(obs['obstacle_voxels'])}")

        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if (i+1) % 20 == 0:
                print(
                    f"Step {i+1}: Reward={reward:.2f}, Term={terminated}, Trunc={truncated}, Dist={info.get('distance_to_target', -1):.2f}")
            if terminated or truncated:
                print(f"Episode finished at step {i+1}.")
                obs, info = env.reset()
                print("Environment reset after episode finish.")
                print(f"New Voxel grid sum: {np.sum(obs['obstacle_voxels'])}")
    except Exception as e:
        import traceback
        print(f"Error during FrankaShelfEnv example usage: {e}")
        traceback.print_exc()
    finally:
        if env:
            env.close()
