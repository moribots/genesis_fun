"""
Provides a dedicated API wrapper for the Genesis simulation engine.

This class encapsulates all direct interactions with the `genesis` library,
offering a simplified and well-defined interface for creating scenes,
managing entities (robots, objects), controlling actuators, and querying
simulation state. By isolating Genesis-specific calls here, the main RL
environment logic remains simulator-agnostic, facilitating easier migration
to other physics engines like Isaac Lab.
"""
from typing import Tuple, List, Any, Optional, Dict, Union

import genesis as gs
import torch
import numpy as np


def to_numpy(data: Union[torch.Tensor, np.ndarray, List, Tuple]) -> np.ndarray:
    """
    Converts input data to a NumPy array.

    Handles PyTorch Tensors (including moving them from CUDA to CPU), lists,
    and tuples.

    Args:
        data: The input data to convert. Can be a PyTorch Tensor, NumPy array,
              list, or tuple.

    Returns:
        The data converted to a NumPy array with a float32 dtype.
    """
    if isinstance(data, torch.Tensor):
        # Move tensor to CPU before converting to numpy if it's on a CUDA device
        return data.cpu().numpy() if data.is_cuda else data.numpy()
    return np.asarray(data, dtype=np.float32)


class GenesisAPI:
    """A wrapper class that encapsulates all Genesis simulation API calls.

    This class manages the simulation lifecycle, including initialization,
    scene creation, stepping, and cleanup. It provides a clean, high-level
    interface for the environment to interact with the Genesis simulator.
    """

    def __init__(self, dt: float, render: bool, device: str):
        """
        Initializes the Genesis API wrapper.

        Note: This does not initialize the Genesis backend itself. Call the
        `initialize` method for that.

        Args:
            dt: The simulation time step (in seconds).
            render: If True, the simulation will be rendered in a graphical viewer.
            device: The compute device to use ('cpu' or 'cuda').
        """
        self.dt = dt
        self.render = render
        self.device = device
        self.scene: Optional[gs.Scene] = None
        self._is_initialized = False
        self._is_recording = False
        self.video_capture_camera: Optional[Any] = None
        self.plane_entity: Optional[Any] = None
        self.franka_entity: Optional[Any] = None
        self.shelf_component_entities: List[Any] = []
        self.target_sphere_entity_type: Optional[Any] = None

    def initialize(self):
        """Initializes the Genesis simulation backend if not already done."""
        # Check if Genesis has already been initialized globally
        if not (hasattr(gs, '_is_initialized') and gs._is_initialized):
            print(f"Initializing Genesis with backend: {self.device.upper()}")
            backend_to_use = gs.gpu if self.device == 'cuda' else gs.cpu
            gs.init(backend=backend_to_use)
            self._is_initialized = True

    def create_scene(self,
                     num_envs: int,
                     env_spacing: Tuple[float, float],
                     video_camera_params: Dict[str, Any],
                     franka_xml_path: str,
                     shelf_component_sizes: List[np.ndarray],
                     include_shelf: bool) -> None:
        """
        Creates the simulation scene with all required actors and sensors.

        This method sets up the simulation world, including the ground plane,
        robot, shelf, and target visualizations. It also configures the camera
        for video recording.

        Args:
            num_envs: The number of parallel environments to create.
            env_spacing: The (x, y) spacing between adjacent environments.
            video_camera_params: A dictionary containing camera settings:
                - 'pos': (x, y, z) position of the camera.
                - 'lookat': (x, y, z) point for the camera to look at.
                - 'fov': Field of view in degrees.
                - 'res': (width, height) resolution of the camera.
            franka_xml_path: Filesystem path to the Franka robot's MJCF XML file.
            shelf_component_sizes: A list of NumPy arrays, where each array
                                   defines the (x, y, z) size of a shelf component.
            include_shelf: If True, the shelf is added to the scene.
        """
        # Configure viewer and simulation options
        sim_viewer_options = gs.options.ViewerOptions(
            camera_pos=video_camera_params['pos'],
            camera_lookat=video_camera_params['lookat'],
            camera_fov=video_camera_params['fov'],
            res=video_camera_params['res'],
            max_FPS=60
        )
        sim_options = gs.options.SimOptions(dt=self.dt)

        # Create the main scene object
        self.scene = gs.Scene(
            viewer_options=sim_viewer_options,
            sim_options=sim_options,
            show_viewer=self.render
        )

        # Add a dedicated camera for video recording
        self.video_capture_camera = self.scene.add_camera(
            res=video_camera_params['res'],
            pos=video_camera_params['pos'],
            lookat=video_camera_params['lookat'],
            fov=video_camera_params['fov'],
            GUI=False
        )

        # Add primary actors to the scene
        self.plane_entity = self.scene.add_entity(gs.morphs.Plane())
        self.franka_entity = self.scene.add_entity(
            gs.morphs.MJCF(file=franka_xml_path))

        # Add shelf components if required
        if include_shelf:
            self.shelf_component_entities = [
                self.scene.add_entity(gs.morphs.Box(
                    pos=(0, -10 - i * 0.5, 0),  # Initial off-screen position
                    quat=(1, 0, 0, 0),
                    size=tuple(s),
                    fixed=True,
                    collision=True,
                    visualization=True
                )) for i, s in enumerate(shelf_component_sizes)
            ]

        # Add a single entity type for target spheres for efficient rendering
        if self.scene and num_envs > 1:
            self.target_sphere_entity_type = self.scene.add_entity(gs.morphs.Sphere(
                pos=(0, -20, 0),  # Initial off-screen position
                radius=0.03,
                visualization=True,
                collision=False,
                fixed=True)
            )

        # Finalize the scene build for parallel environments
        if not self.scene.is_built:
            self.scene.build(n_envs=num_envs, env_spacing=env_spacing)

    def step(self):
        """Advances the simulation by one time step and renders video frame if recording."""
        if self.scene:
            self.scene.step()
            # If recording, capture the frame from the dedicated camera
            if self._is_recording:
                self.video_capture_camera.render()

    def close(self):
        """Shuts down the Genesis scene and the simulation backend."""
        # Safely close the scene
        if self.scene:
            try:
                # Use the appropriate shutdown method based on Genesis version
                if hasattr(self.scene, 'close') and callable(self.scene.close):
                    self.scene.close()
                elif hasattr(self.scene, 'shutdown') and callable(self.scene.shutdown):
                    self.scene.shutdown()
            finally:
                self.scene = None

        # Shut down the global Genesis instance
        if self._is_initialized and hasattr(gs, 'shutdown') and callable(gs.shutdown):
            gs.shutdown()
            self._is_initialized = False

    def start_video_recording(self):
        """Starts video recording using the dedicated capture camera."""
        if self.video_capture_camera:
            self._is_recording = True
            self.video_capture_camera.start_recording()
            print("Video recording started.")

    def stop_video_recording(self, file_path: str):
        """Stops video recording and saves the output to a file.

        Args:
            file_path: The full path (including filename) to save the video.
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

    def draw_debug_spheres(self, points: List, radius: float, color: Tuple):
        """Draws debugging spheres in the scene (for single-env visualization).

        Args:
            points: A list of [x, y, z] coordinates for the sphere centers.
            radius: The radius of the spheres.
            color: A (r, g, b, a) tuple for the sphere color.
        """
        if self.scene:
            self.scene.draw_debug_spheres(points, radius=radius, color=color)

    def set_target_sphere_positions(self, positions: np.ndarray):
        """Sets the positions of the target visualization spheres for all environments.

        Args:
            positions: A NumPy array of shape (num_envs, 3) with the target positions.
        """
        if self.target_sphere_entity_type is not None:
            pos_tensor = torch.from_numpy(positions).to(
                self.device, dtype=torch.float32)
            self.target_sphere_entity_type.set_pos(pos_tensor)

    def get_franka_dof_indices(self, joint_names: List[str]) -> np.ndarray:
        """Retrieves the local DOF (Degree of Freedom) indices for specified joint names.

        Args:
            joint_names: A list of joint names to query.

        Returns:
            A NumPy array of integer DOF indices.
        """
        return np.array([self.franka_entity.get_joint(name).dof_idx_local for name in joint_names], dtype=np.int32)

    def get_franka_ee_link_index(self, link_name: str) -> int:
        """Retrieves the global index for a specified link name.

        Args:
            link_name: The name of the link to query (e.g., "hand").

        Returns:
            The integer index of the link.
        """
        return self.franka_entity.get_link(link_name).idx

    def get_dof_positions(self, dof_indices: np.ndarray) -> np.ndarray:
        """Gets the current positions (qpos) for the specified DOFs.

        Args:
            dof_indices: A NumPy array of DOF indices to query.

        Returns:
            A NumPy array of shape (num_envs, num_dofs) with the joint positions.
        """
        return to_numpy(self.franka_entity.get_dofs_position(dof_indices))

    def get_dof_velocities(self, dof_indices: np.ndarray) -> np.ndarray:
        """Gets the current velocities (qvel) for the specified DOFs.

        Args:
            dof_indices: A NumPy array of DOF indices to query.

        Returns:
            A NumPy array of shape (num_envs, num_dofs) with the joint velocities.
        """
        return to_numpy(self.franka_entity.get_dofs_velocity(dof_indices))

    def get_ee_state(self, ee_link_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gets the full state of the end-effector.

        Args:
            ee_link_name: The name of the end-effector link.

        Returns:
            A tuple containing:
            - ee_pos (np.ndarray): Position (x, y, z) of shape (num_envs, 3).
            - ee_quat_wxyz (np.ndarray): Orientation quaternion (w, x, y, z)
              of shape (num_envs, 4).
            - ee_vel (np.ndarray): 6D velocity (linear_x,y,z, angular_x,y,z)
              of shape (num_envs, 6).
        """
        # Get position and orientation directly from the link
        ee_link = self.franka_entity.get_link(ee_link_name)
        ee_pos = to_numpy(ee_link.get_pos())
        ee_quat_wxyz = to_numpy(ee_link.get_quat())

        # Get velocities for all links and then select the end-effector's
        all_links_vel = self.franka_entity.get_links_vel()
        all_links_ang = self.franka_entity.get_links_ang()
        ee_link_idx = self.get_franka_ee_link_index(ee_link_name)
        ee_linear_vel = all_links_vel[:, ee_link_idx, :]
        ee_angular_vel = all_links_ang[:, ee_link_idx, :]

        # Combine linear and angular velocities into a single 6D tensor
        ee_vel_tensor = torch.cat([ee_linear_vel, ee_angular_vel], dim=1)
        ee_vel = to_numpy(ee_vel_tensor)

        return ee_pos, ee_quat_wxyz, ee_vel

    def get_collisions(self) -> np.ndarray:
        """Checks for any collisions involving the Franka robot.

        Returns:
            A boolean NumPy array of shape (num_envs,) where `True` indicates
            a collision occurred in that environment.
        """
        if self.scene and self.scene.is_built and self.franka_entity:
            contacts = self.franka_entity.get_contacts()
            # Ensure contacts data is valid before processing
            if contacts and 'valid_mask' in contacts:
                valid_mask = to_numpy(contacts['valid_mask'])
                # Return True for any env where at least one contact is valid
                return np.any(valid_mask, axis=1)
        # Default to no collisions if scene is not ready
        return np.zeros(self.scene.n_envs if self.scene else 0, dtype=bool)

    def set_dof_state(self, positions: np.ndarray, velocities: np.ndarray, dof_indices: np.ndarray, env_ids: List[int]):
        """Sets the position and velocity for specified DOFs in specified environments.

        Args:
            positions: A NumPy array of shape (len(env_ids), num_dofs) for target positions.
            velocities: A NumPy array of shape (len(env_ids), num_dofs) for target velocities.
            dof_indices: The DOF indices to modify.
            env_ids: A list of environment indices to modify.
        """
        # Get the current full state tensors
        all_qpos = self.franka_entity.get_dofs_position(dof_indices)
        all_qvel = self.franka_entity.get_dofs_velocity(dof_indices)

        # Convert inputs to tensors
        qpos_tensor = torch.tensor(
            positions, device=self.device, dtype=torch.float32)
        qvel_tensor = torch.tensor(
            velocities, device=self.device, dtype=torch.float32)

        # Update the specific environment slices in the full tensors
        all_qpos[env_ids] = qpos_tensor
        all_qvel[env_ids] = qvel_tensor

        # Apply the updated full state tensors back to the simulation
        self.franka_entity.set_dofs_position(all_qpos, dof_indices)
        self.franka_entity.set_dofs_velocity(all_qvel, dof_indices)

    def set_shelf_component_poses(self, positions: np.ndarray, orientations_wxyz: np.ndarray, component_idx: int, env_ids: List[int]):
        """Sets the pose for a specific shelf component in specified environments.

        Args:
            positions: A NumPy array of shape (len(env_ids), 3) for target positions.
            orientations_wxyz: A NumPy array of shape (len(env_ids), 4) for target orientations.
            component_idx: The index of the shelf component to modify.
            env_ids: A list of environment indices to modify.
        """
        component_entity = self.shelf_component_entities[component_idx]
        # Get the current full pose tensors
        all_pos = component_entity.get_pos()
        all_quat = component_entity.get_quat()

        # Convert inputs to tensors
        pos_tensor = torch.tensor(
            positions, device=self.device, dtype=torch.float32)
        quat_tensor = torch.tensor(
            orientations_wxyz, device=self.device, dtype=torch.float32)

        # Update the specific environment slices in the full tensors
        all_pos[env_ids] = pos_tensor
        all_quat[env_ids] = quat_tensor

        # Apply the updated full pose tensors back to the simulation
        component_entity.set_pos(all_pos)
        component_entity.set_quat(all_quat)

    def apply_velocity_control(self, targets: torch.Tensor, dof_indices: np.ndarray):
        """Applies velocity targets to specified robot DOFs.

        Args:
            targets: A Torch tensor of shape (num_envs, num_dofs) with target velocities.
            dof_indices: The DOF indices to apply control to.
        """
        self.franka_entity.control_dofs_velocity(targets, dof_indices)

    def apply_torque_control(self, torques: torch.Tensor, dof_indices: np.ndarray):
        """Applies torques to specified robot DOFs.

        Args:
            torques: A Torch tensor of shape (num_envs, num_dofs) with target torques.
            dof_indices: The DOF indices to apply control to.
        """
        self.franka_entity.control_dofs_force(torques, dof_indices)
