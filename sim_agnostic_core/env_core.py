"""
Defines the abstract base class for a vectorized reinforcement learning
environment. This ensures that all simulator-specific implementations adhere to a
consistent interface required by the RSL-RL training logic.
"""
import abc
from typing import Tuple, Dict, Optional, List, Union

import torch
import numpy as np
from rsl_rl.env import VecEnv


class BaseFrankaEnv(VecEnv):
    """
    An abstract base class for Franka environments.

    This class provides a common, simulator-agnostic interface for various
    physics simulators (e.g., Genesis, Isaac Sim). It inherits from RSL-RL's
    `VecEnv` and defines the core methods and properties that the training
    runner expects, such as `step`, `reset`, and observation/action space
    definitions.

    Concrete subclasses must implement the abstract methods to provide
    simulator-specific functionality.
    """

    def __init__(self, num_envs: int, num_obs: int, num_actions: int, max_episode_length: int, device: str):
        """
        Initializes the abstract vectorized environment.

        Args:
            num_envs: The number of parallel environments to simulate.
            num_obs: The dimensionality of the observation space.
            num_actions: The dimensionality of the action space.
            max_episode_length: The maximum number of steps per episode before
                                truncation.
            device: The compute device ('cpu' or 'cuda') for torch tensors.
        """
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.max_episode_length = max_episode_length
        self.device = device

        # --- Standard RSL-RL VecEnv required attributes ---
        # Can be overridden by subclasses if privileged observations are used.
        self.num_privileged_obs = None

        # A dictionary for passing extra data (e.g., for logging) from the
        # environment to the runner.
        self.extras = {}

        # Core buffers required by the RSL-RL runner.
        # These are expected to be torch tensors on the specified device.
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
            actions: A tensor of shape (num_envs, num_actions) with actions
                     to apply to each environment.

        Returns:
            A tuple containing:
            - obs_buf: The new observation buffer.
            - privileged_obs_buf: The privileged observation buffer (or None).
            - rew_buf: The reward buffer for the step.
            - reset_buf: The reset buffer (1 for done, 0 otherwise).
            - extras: A dictionary of extra information for logging.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """
        Resets all environments to their initial state.

        Returns:
            A tuple containing the initial observation buffer and the extras dict.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset_idx(self, env_ids: Union[np.ndarray, List[int]]):
        """
        Resets specific environments by their indices.

        Args:
            env_ids: A list or numpy array of environment indices to reset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Cleans up resources used by the environment and the underlying simulator.
        This should be called at the end of the training process.
        """
        raise NotImplementedError

    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """
        Returns the current observation buffer and extras dictionary.
        """
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """
        Returns the privileged observation buffer, if it exists.
        """
        return self.num_privileged_obs

    # --- Optional methods for utilities like video recording ---

    def start_video_recording(self):
        """
        Optional method to start video recording.
        Subclasses should implement this if they support video capture.
        """
        pass

    def stop_video_recording(self, file_path: str):
        """
        Optional method to stop video recording and save it.
        Subclasses should implement this if they support video capture.

        Args:
            file_path (str): The path to save the video file.
        """
        pass

    def get_episode_infos(self) -> list:
        """
        Optional method to return a list of episode info dictionaries.
        This is typically called by the runner for logging episode stats.
        An info dict usually contains 'reward' and 'length'.
        """
        return []
