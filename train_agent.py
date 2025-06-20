"""
Main entry point for training a PPO agent on the Franka environment.

This script orchestrates the entire training process:
- Parses and sets up configuration using dataclasses.
- Initializes Weights & Biases for logging, if enabled.
- Instantiates the concrete simulation environment (`GenesisFrankaEnv`).
- Creates and runs the `CustomOnPolicyRunner`, which manages the training loop,
  data collection, and learning updates.
- Handles cleanup of the environment and simulator upon completion or error.
"""
import os
import sys
import traceback
from dataclasses import asdict

import torch
import wandb

from sim_agnostic_core.training_core import TrainConfig, CustomOnPolicyRunner
# Import the concrete Genesis environment implementation. To switch simulators,
# you would change this import to point to the new environment bridge,
# e.g., `from isaac_bridge.isaac_env import IsaacFrankaEnv`.
from genesis_bridge.genesis_env import GenesisFrankaEnv


def setup_wandb(cfg: TrainConfig):
    """
    Initializes and configures a Weights & Biases run.

    Args:
        cfg: The master training configuration object.

    Returns:
        The initialized wandb run object, or None if W&B is disabled or fails.
    """
    if not cfg.runner.wandb:
        return None

    try:
        # Attempt to log in if no API key is found
        if wandb.api.api_key is None:
            wandb.login()

        # If entity is not specified, fetch the default entity associated with the API key
        if cfg.runner.wandb_entity is None:
            api = wandb.Api()
            cfg.runner.wandb_entity = api.default_entity
            if not cfg.runner.wandb_entity:
                raise ValueError(
                    "Could not determine W&B default entity. Please specify it.")
            print(
                f"--- W&B entity not provided, using default: {cfg.runner.wandb_entity} ---")

        # Initialize the W&B run
        wandb_run = wandb.init(
            project=cfg.runner.wandb_project,
            entity=cfg.runner.wandb_entity,
            group=cfg.runner.wandb_group or None,
            name=cfg.runner.run_name or None,
            tags=cfg.runner.wandb_tags,
            config=asdict(cfg)  # Log the entire configuration
        )
        return wandb_run

    except Exception as e:
        print(f"--- W&B setup failed: {e}. Disabling W&B logging. ---")
        cfg.runner.wandb = False
        if 'wandb_run' in locals() and wandb_run:
            wandb_run.finish()
        return None


def run_franka_training(cfg: TrainConfig) -> None:
    """
    Main orchestrator for the Franka agent training process.

    Args:
        cfg: A TrainConfig object containing all configuration parameters.
    """
    env = None
    wandb_run = None

    try:
        # --- 1. Initialize W&B ---
        wandb_run = setup_wandb(cfg)

        # --- 2. Create Environment ---
        # The environment-specific kwargs are taken directly from the EnvConfig
        # dataclass, making the setup clean and type-safe.
        env_kwargs = asdict(cfg.env)
        # Add runner-specific device option, as the env needs it for torch tensors
        env_kwargs['device'] = cfg.runner.device
        # RSL-RL runner expects this key, so we rename it from the env config
        env_kwargs['max_steps_per_episode'] = env_kwargs.pop(
            'max_episode_length')

        # This is the key point for simulator agnosticism. We instantiate the
        # concrete class imported at the top of the file.
        env = GenesisFrankaEnv(**env_kwargs)
        print("--- GenesisFrankaEnv created successfully. ---")

        # --- 3. Create Log Directory ---
        os.makedirs(cfg.runner.log_dir, exist_ok=True)

        # --- 4. Instantiate and Run the Custom Runner ---
        # The runner is fully sim-agnostic and works with any environment that
        # adheres to the BaseFrankaEnv interface.
        runner = CustomOnPolicyRunner(
            env, cfg, cfg.runner.log_dir, device=cfg.runner.device)
        print("--- CustomOnPolicyRunner instantiated. Starting training... ---")

        runner.learn(
            num_learning_iterations=cfg.runner.max_iterations,
            init_at_random_ep_len=True
        )
        print("--- Training finished successfully. ---")

    except Exception as e:
        print(
            f"\n!!! An unhandled exception occurred during training: {e} !!!")
        traceback.print_exc()
        sys.exit(1)

    finally:
        # --- 5. Cleanup ---
        print("\n--- Performing cleanup... ---")
        if env is not None:
            env.close()
            print("Environment and simulator closed.")
        if wandb_run:
            wandb_run.finish()
            print("W&B run finished.")
        print("--- Training script finished. ---")


if __name__ == '__main__':
    # Create the top-level configuration object.
    # This will automatically initialize all nested dataclasses with their
    # default values and run the `__post_init__` methods.
    training_configuration = TrainConfig()

    # Start the training process
    run_franka_training(training_configuration)
