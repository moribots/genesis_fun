import unittest
import torch
import gymnasium as gym  # Use gymnasium
from gymnasium import spaces
import numpy as np

# Assuming collision_cnn.py is in the same directory or accessible via PYTHONPATH
# This will import the REVISED CustomCNNFeatureExtractor structure
from collision_cnn import CustomCNNFeatureExtractor


class TestCustomCNNFeatureExtractor(unittest.TestCase):
    """
    Unit tests for the CustomCNNFeatureExtractor class,
    updated for the revised architecture.
    """

    def _create_mock_observation_space(self, voxel_shape=(32, 48, 48), other_obs_dim=10):
        """
        Helper to create a mock observation space.
        Voxel shape: (Depth, Height, Width) for 3D or (Channels, Depth, Height, Width) for 4D.
        Using recommended (32, 48, 48) as default for 3D.
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

    def test_initialization_3d_voxels_revised_arch(self):
        """
        Tests initialization with 3D voxel input (D, H, W) for the revised architecture.
        """
        voxel_shape_3d = (32, 48, 48)  # Recommended D, H, W
        features_dim_out = 256
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_3d)

        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=features_dim_out)

        self.assertIsInstance(extractor.cnn_base, torch.nn.Sequential)
        self.assertIsInstance(extractor.pool, torch.nn.AdaptiveAvgPool3d)
        self.assertIsInstance(extractor.flatten, torch.nn.Flatten)
        self.assertIsInstance(extractor.linear, torch.nn.Linear)

        # _features_dim should now be equal to features_dim_out due to the final linear layer
        self.assertEqual(extractor._features_dim, features_dim_out)
        self.assertEqual(extractor.linear.out_features, features_dim_out)

        # Verify input channels for Conv3d in cnn_base
        self.assertEqual(extractor.cnn_base[0].in_channels, 1)

    def test_initialization_4d_voxels_revised_arch(self):
        """
        Tests initialization with 4D voxel input (C, D, H, W) for the revised architecture.
        """
        voxel_shape_4d = (3, 32, 48, 48)  # C, D, H, W
        features_dim_out = 128
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_4d)

        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=features_dim_out)

        self.assertIsInstance(extractor.cnn_base, torch.nn.Sequential)
        self.assertIsInstance(extractor.pool, torch.nn.AdaptiveAvgPool3d)
        self.assertIsInstance(extractor.linear, torch.nn.Linear)
        self.assertEqual(extractor._features_dim, features_dim_out)
        self.assertEqual(extractor.linear.out_features, features_dim_out)

        # Channels should match
        self.assertEqual(extractor.cnn_base[0].in_channels, voxel_shape_4d[0])

    def test_forward_pass_3d_voxels_revised_arch(self):
        """
        Tests the forward pass with 3D voxel input (D, H, W) for the revised architecture.
        Checks output shape and type.
        """
        batch_size = 4
        voxel_shape_3d = (32, 48, 48)  # Recommended D, H, W
        features_dim_out = 64
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_3d)
        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=features_dim_out)

        mock_voxel_data = torch.rand(batch_size, *voxel_shape_3d)
        mock_other_data = torch.rand(
            batch_size, observation_space["other_input"].shape[0])
        mock_obs_dict = {
            "obstacle_voxels": mock_voxel_data,
            "other_input": mock_other_data
        }

        features = extractor(mock_obs_dict)

        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], batch_size)
        # The output features should match features_dim_out
        self.assertEqual(features.shape[1], features_dim_out)

    def test_forward_pass_4d_voxels_revised_arch(self):
        """
        Tests the forward pass with 4D voxel input (C, D, H, W) for the revised architecture.
        Checks output shape and type.
        """
        batch_size = 2
        # C, D, H, W (using 1 channel for simplicity)
        voxel_shape_4d = (1, 32, 48, 48)
        features_dim_out = 32
        observation_space = self._create_mock_observation_space(
            voxel_shape=voxel_shape_4d)
        extractor = CustomCNNFeatureExtractor(
            observation_space, features_dim=features_dim_out)

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
        self.assertEqual(features.shape[1], features_dim_out)

    def test_unsupported_voxel_shape(self):
        """
        Tests that initialization fails with unsupported voxel dimensions.
        """
        voxel_shape_2d = (32, 32)
        with self.assertRaises(ValueError):
            # This error is now caught in _create_mock_observation_space if len is not 3 or 4
            # If it passed that, CustomCNNFeatureExtractor might also raise an error.
            observation_space = self._create_mock_observation_space(
                voxel_shape=voxel_shape_2d)
            # Error might be raised by CustomCNNFeatureExtractor's __init__ if shape check is strict
            # For now, the ValueError in _create_mock_observation_space will be hit first.

        voxel_shape_5d = (1, 3, 16, 32, 32)
        with self.assertRaises(ValueError):
            observation_space = self._create_mock_observation_space(
                voxel_shape=voxel_shape_5d)
            # As above, error from helper or constructor.

    def test_missing_obstacle_voxels_key(self):
        """
        Tests that initialization fails if 'obstacle_voxels' key is missing.
        """
        observation_space = spaces.Dict({
            "some_other_key": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })
        # CustomCNNFeatureExtractor raises AssertionError
        with self.assertRaises(AssertionError):
            CustomCNNFeatureExtractor(observation_space, features_dim=128)

    def test_incorrect_obstacle_voxels_type(self):
        """
        Tests that initialization fails if 'obstacle_voxels' is not a Box space.
        """
        observation_space = spaces.Dict({
            "obstacle_voxels": spaces.Discrete(5)
        })
        # CustomCNNFeatureExtractor raises AssertionError
        with self.assertRaises(AssertionError):
            CustomCNNFeatureExtractor(observation_space, features_dim=128)


if __name__ == '__main__':
    # Note: You'll need to have your revised collision_cnn.py available for this to run.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
