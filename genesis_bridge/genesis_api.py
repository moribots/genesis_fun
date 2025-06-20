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
    Converts input data (PyTorch Tensor, list, tuple) to a NumPy array.
    If the input is a PyTorch Tensor on a CUDA device, it's moved to the CPU.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy() if data.is_cuda else data.numpy()
    return np.asarray(data, dtype=np.float32)


class GenesisAPI:
    """A wrapper for the Genesis simulation engine API."""

    def __init__(self, dt: float, render: bool, device: str):
        """
        Initializes the Genesis simulator.

        Args:
            dt: The simulation time step.
            render: Whether to show the graphical viewer.
            device: The compute device ('cpu' or 'cuda').
        """
        self.dt = dt
        self.render = render
        self.device = device
        self.scene: Optional[gs.Scene] = None
        self._is_initialized = False

    def initialize(self):
        """Initializes the Genesis backend."""
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
        Creates the simulation scene with all required entities.
        """
        sim_viewer_options = gs.options.ViewerOptions(
            camera_pos=video_camera_params['pos'],
            camera_lookat=video_camera_params['lookat'],
            camera_fov=video_camera_params['fov'],
            res=video_camera_params['res'],
            max_FPS=60
        )
        sim_options = gs.options.SimOptions(dt=self.dt)
        self.scene = gs.Scene(
            viewer_options=sim_viewer_options,
            sim_options=sim_options,
            show_viewer=self.render
        )

        # Add camera for video recording
        self.video_capture_camera = self.scene.add_camera(
            res=video_camera_params['res'],
            pos=video_camera_params['pos'],
            lookat=video_camera_params['lookat'],
            fov=video_camera_params['fov'],
            GUI=False
        )
        self._is_recording = False

        # Add entities
        self.plane_entity = self.scene.add_entity(gs.morphs.Plane())
        self.franka_entity = self.scene.add_entity(
            gs.morphs.MJCF(file=franka_xml_path))

        self.shelf_component_entities: List[Any] = []
        if include_shelf:
            self.shelf_component_entities = [self.scene.add_entity(gs.morphs.Box(
                pos=(0, -10 - i * 0.5, 0), quat=(1, 0, 0, 0), size=tuple(s),
                fixed=True, collision=True, visualization=True
            )) for i, s in enumerate(shelf_component_sizes)]

        self.target_sphere_entity_type: Optional[Any] = None
        if self.scene and num_envs > 1:
            self.target_sphere_entity_type = self.scene.add_entity(gs.morphs.Sphere(
                pos=(0, -20, 0), radius=0.03, visualization=True, collision=False, fixed=True))

        # Build the scene for parallel environments
        if not self.scene.is_built:
            self.scene.build(n_envs=num_envs, env_spacing=env_spacing)

    def step(self):
        """Advances the simulation by one time step."""
        if self.scene:
            self.scene.step()
            if self._is_recording:
                self.video_capture_camera.render()

    def close(self):
        """Shuts down the Genesis simulator."""
        if self.scene:
            try:
                if hasattr(self.scene, 'close') and callable(self.scene.close):
                    self.scene.close()
                elif hasattr(self.scene, 'shutdown') and callable(self.scene.shutdown):
                    self.scene.shutdown()
            finally:
                self.scene = None
        if self._is_initialized and hasattr(gs, 'shutdown') and callable(gs.shutdown):
            gs.shutdown()

    def start_video_recording(self):
        if self.video_capture_camera:
            self._is_recording = True
            self.video_capture_camera.start_recording()
            print("Video recording started.")

    def stop_video_recording(self, file_path: str):
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
        if self.scene:
            self.scene.draw_debug_spheres(points, radius=radius, color=color)

    def set_target_sphere_positions(self, positions: np.ndarray):
        if self.target_sphere_entity_type is not None:
            pos_tensor = torch.from_numpy(positions).to(
                self.device, dtype=torch.float32)
            self.target_sphere_entity_type.set_pos(pos_tensor)

    def get_franka_dof_indices(self, joint_names: List[str]) -> np.ndarray:
        return np.array([self.franka_entity.get_joint(name).dof_idx_local for name in joint_names], dtype=np.int32)

    def get_franka_ee_link_index(self, link_name: str) -> int:
        return self.franka_entity.get_link(link_name).idx

    def get_dof_positions(self, dof_indices: np.ndarray) -> np.ndarray:
        return to_numpy(self.franka_entity.get_dofs_position(dof_indices))

    def get_dof_velocities(self, dof_indices: np.ndarray) -> np.ndarray:
        return to_numpy(self.franka_entity.get_dofs_velocity(dof_indices))

    def get_ee_state(self, ee_link_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ee_link = self.franka_entity.get_link(ee_link_name)
        ee_pos = to_numpy(ee_link.get_pos())
        ee_quat_wxyz = to_numpy(ee_link.get_quat())

        all_links_vel = self.franka_entity.get_links_vel()
        all_links_ang = self.franka_entity.get_links_ang()
        ee_link_idx = self.get_franka_ee_link_index(ee_link_name)
        ee_linear_vel = all_links_vel[:, ee_link_idx, :]
        ee_angular_vel = all_links_ang[:, ee_link_idx, :]

        ee_vel_tensor = torch.cat([ee_linear_vel, ee_angular_vel], dim=1)
        ee_vel = to_numpy(ee_vel_tensor)

        return ee_pos, ee_quat_wxyz, ee_vel

    def get_collisions(self) -> np.ndarray:
        if self.scene and self.scene.is_built and self.franka_entity:
            contacts = self.franka_entity.get_contacts()
            if contacts and 'valid_mask' in contacts:
                valid_mask = to_numpy(contacts['valid_mask'])
                return np.any(valid_mask, axis=1)
        return np.zeros(self.scene.n_envs if self.scene else 0, dtype=bool)

    def set_dof_state(self, positions: np.ndarray, velocities: np.ndarray, dof_indices: np.ndarray, env_ids: List[int]):
        all_qpos = self.franka_entity.get_dofs_position(dof_indices)
        all_qvel = self.franka_entity.get_dofs_velocity(dof_indices)

        qpos_tensor = torch.tensor(
            positions, device=self.device, dtype=torch.float32)
        qvel_tensor = torch.tensor(
            velocities, device=self.device, dtype=torch.float32)

        all_qpos[env_ids] = qpos_tensor
        all_qvel[env_ids] = qvel_tensor

        self.franka_entity.set_dofs_position(all_qpos, dof_indices)
        self.franka_entity.set_dofs_velocity(all_qvel, dof_indices)

    def set_shelf_component_poses(self, positions: np.ndarray, orientations_wxyz: np.ndarray, component_idx: int, env_ids: List[int]):
        component_entity = self.shelf_component_entities[component_idx]
        all_pos = component_entity.get_pos()
        all_quat = component_entity.get_quat()

        pos_tensor = torch.tensor(
            positions, device=self.device, dtype=torch.float32)
        quat_tensor = torch.tensor(
            orientations_wxyz, device=self.device, dtype=torch.float32)

        all_pos[env_ids] = pos_tensor
        all_quat[env_ids] = quat_tensor

        component_entity.set_pos(all_pos)
        component_entity.set_quat(all_quat)

    def apply_velocity_control(self, targets: torch.Tensor, dof_indices: np.ndarray):
        self.franka_entity.control_dofs_velocity(targets, dof_indices)

    def apply_torque_control(self, torques: torch.Tensor, dof_indices: np.ndarray):
        self.franka_entity.control_dofs_force(torques, dof_indices)
