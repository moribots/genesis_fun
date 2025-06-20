# Simulator-Agnostic Franka RL Training

This project provides a framework for training a Franka Emika Panda robot for trajectory-tracking tasks using reinforcement learning. The key feature of this repository is its simulator-agnostic architecture, which completely decouples the core RL logic from the underlying physics simulator.

Currently, it provides a concrete implementation for the [Genesis simulator](https://genesis-world.readthedocs.io/en/latest/), but it is structured to be easily extended to other simulators like NVIDIA Isaac Lab.

## Project Structure


```
genesis_fun/
├── sim_agnostic_core/
│   ├── __init__.py
│   ├── curriculum_core.py
│   ├── env_core.py
│   └── training_core.py
├── genesis_bridge/
│   ├── __init__.py
│   ├── genesis_api.py
│   └── genesis_env.py
├── train_agent.py
├── requirements.txt
├── .gitignore
└── ... (other existing project files)
```

The codebase is organized into two main packages, which separate the core logic from the simulator-specific implementation:

* `sim_agnostic_core/`: This directory contains all the high-level reinforcement learning logic that is independent of any specific simulator.

  * `training_core.py`: Defines the configuration dataclasses and a custom RSL-RL `OnPolicyRunner` for handling the training loop, logging, and model checkpointing.

  * `env_core.py`: Defines the abstract `BaseFrankaEnv` class, which specifies the required interface that all simulator-specific environments must implement.

  * `curriculum_core.py`: Contains the logic for curriculum learning, allowing for dynamic adjustment of task difficulty and reward parameters based on agent performance.

* `genesis_bridge/`: This directory acts as the bridge to the Genesis simulator. It contains all the code that directly interacts with the Genesis API.

  * `genesis_api.py`: A dedicated wrapper class that encapsulates all calls to the `genesis` library, providing a clean interface for scene creation, state querying, and robot control.

  * `genesis_env.py`: The concrete implementation of the `BaseFrankaEnv`, which uses the `GenesisAPI` to run the simulation.

This modular design allows you to develop and test your RL algorithms independently of the simulation backend. To port this project to a new simulator, you would only need to create a new "bridge" package (e.g., `isaac_lab_bridge`) that implements the required interfaces.

## Setup and Installation

Follow these steps to set up the project and install the necessary dependencies.

### 1. Prerequisites

* Python 3.8 or later.

* If you have a compatible NVIDIA GPU, ensure that you have the appropriate CUDA drivers installed to leverage GPU acceleration with PyTorch and Genesis.

### 2. Install Dependencies

With a virtual environment (ideally conda), install the required Python packages using the `requirements.txt` file:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

```

TODO: install instructions for RSL-RL and Genesis

## Running the Training

The main entry point for training the agent is the `train_agent.py` script.

### Basic Training Command

To start a training session with the default configuration (torque control, logging to W&B), simply run:

```
python train_agent.py

```

### Configuration

The training process is configured using dataclasses defined in `sim_agnostic_core/training_core.py`. You can modify the `TrainConfig` object in `train_agent.py` to change hyperparameters, environment settings, or runner options.

Key configuration options include:

* **Control Mode**: In `training_core.py`, you can change `control_mode` within the `EnvConfig` dataclass from `'torque'` to `'velocity'` to switch the robot's control paradigm. The reward functions and curriculum settings will adjust automatically.

* **Logging**: By default, the runner logs metrics and videos to [Weights & Biases](https://wandb.ai).

  * You will be prompted to log in to your W&B account the first time you run the script.

  * You can customize the W&B project, entity, and run name in the `RunnerConfig` dataclass.

  * To disable W&B logging, set `wandb: bool = False` in `RunnerConfig`.

### Checkpoints and Videos

* Training checkpoints (model weights) will be saved periodically to the `./training_logs/` directory.

* Videos of the policy's performance will be logged to W&B at regular intervals, as defined by `video_log_interval` in the `RunnerConfig`.

## Extending to a New Simulator

To adapt this framework for a new simulator like Isaac Lab, you would need to:

1. Create a new directory (e.g., `isaac_lab_bridge`).

2. Inside this new directory, create an `IsaacAPI.py` file that wraps the new simulator's API calls.

3. Create an `IsaacEnv.py` file with a class `IsaacFrankaEnv` that inherits from `BaseFrankaEnv` and uses your new `IsaacAPI` to implement the required `step`, `reset`, and state-querying methods.

4. In `train_agent.py`, change the import from `genesis_bridge.genesis_env` to your new `isaac_lab_bridge.isaac_env` and instantiate `IsaacFrankaEnv` instead of `GenesisFrankaEnv`.

All of the core training logic, curriculum, and reward structures will remain unchanged.

