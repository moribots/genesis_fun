import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for processing 3D voxel grid observations.

    This extractor is designed to take a 3D voxel grid (representing obstacles or
    other spatial information) from the observation space and process it through
    a series of 3D convolutional layers, followed by adaptive pooling and a
    linear layer to output a flat feature vector of a specified dimension.

    :param observation_space: The observation space of the environment.
        It must contain a key "obstacle_voxels" which is a gymnasium.spaces.Box.
    :param features_dim: The number of features to extract and output.
                         This will be the size of the flat feature vector.

    Design:
    - Input Shape: (Batch, Channels, Depth, Height, Width)
    - Convolutional Layers (self.cnn_base):
        - The first Conv3D layer typically uses a stride like (1,2,2) for (D,H,W)
          to reduce height and width dimensions while preserving more resolution
          along the depth dimension initially, which can be useful if depth
          represents a distinct axis like distance or a specific orientation.
        - Subsequent Conv3D layers use strides like (2,2,2) to progressively
          reduce all spatial dimensions.
        - The number of channels (e.g., 32 -> 64 -> 128) typically increases
          with network depth to capture more complex features.
        - ReLU activation is used after each convolutional layer for non-linearity.
    - Adaptive Average Pooling (self.pool):
        - `nn.AdaptiveAvgPool3d((target_D, target_H, target_W))` is used to ensure
          that the output of the pooling operation always has fixed spatial
          dimensions (e.g., (2,2,2)), regardless of minor variations in the input
          size from the `cnn_base`. This makes the network more robust to
          changes in input voxel grid dimensions if needed for different experiments.
    - Flatten Layer (self.flatten): Converts the 3D pooled feature maps into a
      1D vector.
    - Linear Layer (self.linear): A fully connected layer that projects the
      flattened features to the final desired `features_dim`.
    - `self._features_dim = features_dim`: This attribute is crucial as it informs
      Stable Baselines3 about the output dimension of this feature extractor.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Ensure the 'obstacle_voxels' key is present and is a Box space
        assert "obstacle_voxels" in observation_space.spaces, \
            "Observation space must contain an 'obstacle_voxels' key."
        assert isinstance(observation_space.spaces["obstacle_voxels"], gym.spaces.Box), \
            "'obstacle_voxels' must be a gym.spaces.Box."

        super().__init__(observation_space, features_dim)

        voxel_obs_space = observation_space.spaces["obstacle_voxels"]
        voxel_shape = voxel_obs_space.shape  # Expected (D,H,W) or (C,D,H,W)

        if len(voxel_shape) == 3:  # (D, H, W)
            in_channels = 1  # Add a channel dimension
        elif len(voxel_shape) == 4:  # (C, D, H, W)
            in_channels = voxel_shape[0]
        else:
            raise ValueError(
                f"Unsupported voxel_shape: {voxel_shape}. Expected 3 or 4 dimensions.")

        # CNN base to extract spatial features
        # Example assumes input D,H,W like (32, 48, 48) as discussed
        self.cnn_base = nn.Sequential(
            # Input: (B, in_channels, D, H, W) e.g. (B, 1, 32, 48, 48)
            nn.Conv3d(in_channels, 32, kernel_size=3,
                      stride=(1, 2, 2), padding=1),
            # Output: (B, 32, 32, 24, 24) (stride on D is 1, H,W are halved)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            # Output: (B, 64, 16, 12, 12)
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1),
            # Output: (B, 128, 8, 6, 6)
            nn.ReLU()
        )

        # Use Adaptive Average Pooling to get a fixed-size output before flattening
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        # Output after pool: (B, 128, 2, 2, 2)

        self.flatten = nn.Flatten()

        # Calculate the flattened size after the pooling layer
        with torch.no_grad():
            if len(voxel_shape) == 3:  # (D,H,W)
                dummy_cnn_input_shape = (1, in_channels) + voxel_shape
            else:  # (C,D,H,W)
                dummy_cnn_input_shape = (1,) + voxel_shape

            dummy_input = torch.zeros(dummy_cnn_input_shape)
            cnn_after_base_output = self.cnn_base(dummy_input)
            pooled_features = self.pool(cnn_after_base_output)
            flattened_size = self.flatten(pooled_features).shape[1]
            # Expected: 128 * 2 * 2 * 2 = 1024 for the example architecture

        # Linear layer to project to the final desired features_dim
        self.linear = nn.Linear(flattened_size, features_dim)

        # This _features_dim attribute is used by Stable Baselines3
        # and should match the output of this extractor
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the observation.

        :param observations: The observation dictionary from the environment.
        :return: The extracted features (a flat vector).
        """
        voxel_data = observations["obstacle_voxels"]

        # Ensure channel dimension exists: (B, D, H, W) -> (B, 1, D, H, W)
        if len(voxel_data.shape) == 4:
            voxel_data = voxel_data.unsqueeze(1)

        x = self.cnn_base(voxel_data)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
