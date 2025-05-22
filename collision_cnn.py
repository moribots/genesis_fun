import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for processing 3D voxel grid observations.

    :param observation_space: The observation space of the environment.
    :param features_dim: Number of features to extract.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Ensure the 'obstacle_voxels' key is present and is a Box space
        assert "obstacle_voxels" in observation_space.spaces, \
            "Observation space must contain an 'obstacle_voxels' key."
        assert isinstance(observation_space.spaces["obstacle_voxels"], gym.spaces.Box), \
            "'obstacle_voxels' must be a gym.spaces.Box."

        # We calculate the output dimension from the CNN part
        # and then add the dimensions of other observation space components.
        # The `features_dim` passed to the constructor will be the final
        # dimension of the features extracted from *all* inputs by MultiInputPolicy's
        # combined extractor. Here, we define the output dim of just the CNN part.
        # This will be concatenated with features from other inputs by SB3.
        # For simplicity, we'll make the CNN output `features_dim` and let SB3 handle
        # concatenation if other inputs are processed by an MLP.
        # However, a more robust way is to calculate the CNN output size precisely.
        # Let's assume `features_dim` is the desired output size for this CNN part.

        super().__init__(observation_space, features_dim)

        voxel_shape = observation_space.spaces["obstacle_voxels"].shape
        # Assuming shape is (Channels, Depth, Height, Width) or (Depth, Height, Width)
        # If (D, H, W), add a channel dimension: (1, D, H, W)
        if len(voxel_shape) == 3:  # D, H, W
            in_channels = 1
            # If you have multiple channels in your voxel grid, adjust accordingly
        elif len(voxel_shape) == 4:  # C, D, H, W
            in_channels = voxel_shape[0]
        else:
            raise ValueError(
                f"Unsupported voxel_shape: {voxel_shape}. Expected 3 or 4 dimensions.")

        # Define a simple 3D CNN architecture
        # Kernel sizes, strides, padding might need tuning based on voxel_grid_dims
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=5,
                      stride=2, padding=2),  # (D/2, H/2, W/2)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2,
                      padding=1),      # (D/4, H/4, W/4)
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1,
                      padding=1),      # (D/4, H/4, W/4)
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened features dimension after the CNN
        # This is crucial for defining the subsequent MLP layers if any, or the final features_dim
        with torch.no_grad():
            # Create a dummy input tensor with the expected shape
            # Batch size of 1, and the rest is the voxel shape
            if len(voxel_shape) == 3:
                dummy_input_shape = (1, 1) + voxel_shape  # (B, C, D, H, W)
            else:  # len(voxel_shape) == 4
                dummy_input_shape = (1,) + voxel_shape  # (B, C, D, H, W)

            dummy_input = torch.zeros(dummy_input_shape)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # The output of this extractor for the 'obstacle_voxels' key will be cnn_output_dim.
        # SB3's MultiInputPolicy will concatenate this with features from other inputs.
        # The `features_dim` in the PPO policy_kwargs refers to the size of the
        # concatenated feature vector *before* it goes into the policy/value MLPs.
        # We need to ensure this extractor's output for 'obstacle_voxels' is correctly sized
        # or that `features_dim` in `super().__init__` is set to this `cnn_output_dim`.
        # For MultiInputPolicy, the `features_dim` in `super().__init__` is often the
        # sum of output dimensions from each sub-extractor (or a shared dimension if using a shared MLP).
        # Here, we are defining an extractor for a *specific key* 'obstacle_voxels'.
        # The `_features_dim` attribute of BaseFeaturesExtractor should be the output dim of *this* extractor.
        self._features_dim = cnn_output_dim

        # If we want an MLP after the CNN within this extractor:
        # self.linear = nn.Sequential(
        #     nn.Linear(cnn_output_dim, features_dim),
        #     nn.ReLU()
        # )
        # self._features_dim = features_dim # If using the MLP above

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the observation.

        :param observations: The observation from the environment, specifically the
                             'obstacle_voxels' part if used within a CombinedExtractor by SB3.
                             If this extractor is directly assigned to a key,
                             `observations` will be `obs_dict['obstacle_voxels']`.
        :return: The extracted features.
        """
        # If observations is a dict (it will be if this is the top-level extractor for MultiInputPolicy)
        # then we need to select the 'obstacle_voxels' key.
        # However, when used with CombinedExtractor, SB3 passes the specific part.
        # Let's assume SB3 passes the correct tensor for 'obstacle_voxels'.
        voxel_data = observations["obstacle_voxels"]

        # Add channel dimension if it's (B, D, H, W) instead of (B, C, D, H, W)
        if len(voxel_data.shape) == 4:  # B, D, H, W
            voxel_data = voxel_data.unsqueeze(1)  # B, 1, D, H, W

        cnn_features = self.cnn(voxel_data)
        # if hasattr(self, 'linear'):
        #    return self.linear(cnn_features)
        return cnn_features
