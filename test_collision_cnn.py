import unittest
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Assuming collision_cnn.py is in the same directory or accessible via PYTHONPATH
from collision_cnn import CustomCNNFeatureExtractor


class TestCustomCNNFeatureExtractor(unittest.TestCase):
    """
    Unit tests for the CustomCNNFeatureExtractor class.
    """

    def _create_mock_observation_space(self, voxel_shape=(1, 16, 32, 32), other_obs_dim=10):
        """
        Helper to create a mock observation space.
        Voxel shape: (Channels, Depth, Height, Width) or (Depth, Height, Width)
        """
        if len(voxel_shape) == 3:  # D, H, W
            obs_space_voxels = spaces.Box(
                low=0, high=1, shape=voxel_shape, dtype=np.float32)
        elif len(voxel_shape) == 4:  # C, D, H, W
            obs_space_voxels = spaces.Box(
                low=0, high=1, shape=voxel_shape, dtype=np.float32)
        else:
            raise ValueError("voxel_shape must have 3 or 4 dimensions")

        return spaces.Dict({
            "obstacle_voxels": obs_space_voxels,
            "other_input": spaces.Box(low=-np.inf, high=np.inf, shape=(other_obs_dim,), dtype=np.float32)
        })

    def test_initialization_3d_voxels(self):
        """
        Tests initialization with 3D voxel input (D, H, W).
        """
        voxel_shape_3d = (16, 32, 32)  # D, H, W
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_3d)

        # The features_dim passed to constructor is for the overall MultiInputPolicy's combined extractor.
        # The CustomCNNFeatureExtractor itself will calculate its own output dimension.
        # Let's test if it calculates _features_dim correctly.
        # features_dim here is a bit misleading for this specific test
        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=256)

        self.assertIsInstance(extractor.cnn, torch.nn.Sequential)
        # Check that _features_dim is set
        self.assertGreater(extractor._features_dim, 0)

        # Verify input channels for Conv3d
        self.assertEqual(extractor.cnn[0].in_channels, 1)

    def test_initialization_4d_voxels(self):
        """
        Tests initialization with 4D voxel input (C, D, H, W).
        """
        voxel_shape_4d = (3, 16, 32, 32)  # C, D, H, W
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_4d)

        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=256)

        self.assertIsInstance(extractor.cnn, torch.nn.Sequential)
        self.assertGreater(extractor._features_dim, 0)
        # Channels should match
        self.assertEqual(extractor.cnn[0].in_channels, voxel_shape_4d[0])

    def test_forward_pass_3d_voxels(self):
        """
        Tests the forward pass with 3D voxel input (D, H, W).
        Checks output shape and type.
        """
        batch_size = 4
        voxel_shape_3d = (16, 32, 32)  # D, H, W
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_3d)
        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=128)

        # Create mock observation data
        mock_voxel_data = torch.rand(batch_size, *voxel_shape_3d)
        mock_other_data = torch.rand(
            batch_size, observation_space["other_input"].shape[0])

        # The extractor expects a dictionary if it's the main one for MultiInputPolicy,
        # or just the tensor for the specific key if used by CombinedExtractor.
        # The current implementation of forward() expects the dict.
        mock_obs_dict = {
            "obstacle_voxels": mock_voxel_data,
            "other_input": mock_other_data  # This won't be used by this specific extractor
        }

        features = extractor(mock_obs_dict)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[1], extractor._features_dim)

    def test_forward_pass_4d_voxels(self):
        """
        Tests the forward pass with 4D voxel input (C, D, H, W).
        Checks output shape and type.
        """
        batch_size = 2
        voxel_shape_4d = (3, 16, 32, 32)  # C, D, H, W
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_4d)
        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=128)

        mock_voxel_data = torch.rand(batch_size, *voxel_shape_4d)
        mock_other_data = torch.rand(
            batch_size, observation_space["other_input"].shape[0])
        mock_obs_dict = {
            "obstacle_voxels": mock_voxel_data,
            "other_input": mock_other_data
        }

        features = extractor(mock_obs_dict)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], batch_size)
        self.assertEqual(features.shape[1], extractor._features_dim)

    def test_unsupported_voxel_shape(self):
        """
        Tests that initialization fails with unsupported voxel dimensions.
        """
        voxel_shape_2d = (32, 32)  # Incorrect shape
        with self.assertRaises(ValueError):
            observation_space = self._create_mock_observation_space(
                voxel_shape=voxel_shape_2d)
            # The error is raised in CustomCNNFeatureExtractor's __init__
            CustomCNNFeatureExtractor(observation_space, features_dim=128)

        voxel_shape_5d = (1, 3, 16, 32, 32)  # Incorrect shape
        with self.assertRaises(ValueError):
            observation_space = self._create_mock_observation_space(
                voxel_shape=voxel_shape_5d)
            CustomCNNFeatureExtractor(observation_space, features_dim=128)

    def test_missing_obstacle_voxels_key(self):
        """
        Tests that initialization fails if 'obstacle_voxels' key is missing.
        """
        observation_space = spaces.Dict({
            "some_other_key": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })
        with self.assertRaises(AssertionError):
            CustomCNNFeatureExtractor(observation_space, features_dim=128)

    def test_incorrect_obstacle_voxels_type(self):
        """
        Tests that initialization fails if 'obstacle_voxels' is not a Box space.
        """
        observation_space = spaces.Dict({
            "obstacle_voxels": spaces.Discrete(5)  # Incorrect type
        })
        with self.assertRaises(AssertionError):
            CustomCNNFeatureExtractor(observation_space, features_dim=128)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
