"""
Defines the abstract base class for a vectorized reinforcement learning
environment, ensuring that all simulator-specific implementations adhere to a
consistent interface required by the training logic.
"""
import abc
from typing import Tuple, Dict, Any, Optional, List, Union

import torch
import numpy as np
from rsl_rl.env import VecEnv


class BaseFrankaEnv(VecEnv):
    """
    An abstract base class for Franka environments, providing a common
    interface for different physics simulators (e.g., Genesis, Isaac Lab).

    This class defines the core methods and properties that the training
    runner expects, such as `step`, `reset`, and observation/action space
    definitions. It leaves the simulator-specific implementation details
    to its concrete subclasses.
    """

    def __init__(self, num_envs: int, num_obs: int, num_actions: int, max_episode_length: int, device: str):
        """
        Initializes the abstract environment.

        Args:
            num_envs: The number of parallel environments.
            num_obs: The dimensionality of the observation space.
            num_actions: The dimensionality of the action space.
            max_episode_length: The maximum number of steps per episode.
            device: The compute device ('cpu' or 'cuda').
        """
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.max_episode_length = max_episode_length
        self.device = device

        # RSL-RL VecEnv required attributes
        self.num_privileged_obs = None  # Can be overridden by subclasses if needed
        self.extras = {}
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Applies an action to the environment and steps the simulation.

        Args:
            actions: A tensor of actions to apply to each environment.

        Returns:
            A tuple containing:
            - obs_buf: The new observation buffer.
            - privileged_obs_buf: The privileged observation buffer (or None).
            - rew_buf: The reward buffer.
            - reset_buf: The reset buffer.
            - extras: A dictionary of extra information.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """
        Resets all environments.

        Returns:
            A tuple containing the initial observation buffer and extras dict.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset_idx(self, env_ids: Union[np.ndarray, List[int]]):
        """
        Resets specific environments by their indices.
        """
        raise NotImplementedError

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """
        Returns the current observation buffer and extras dictionary.
        """
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """
        Returns the privileged observation buffer.
        """
        return self.num_privileged_obs

    @abc.abstractmethod
    def close(self):
        """
        Cleans up resources used by the environment and simulator.
        """
        raise NotImplementedError

    # --- Video Recording ---
    def start_video_recording(self):
        """Optional: Start video recording."""
        pass

    def stop_video_recording(self, file_path: str):
        """Optional: Stop video recording and save."""
        pass

    def get_episode_infos(self) -> list:
        """Optional: returns episode info"""
        return []
