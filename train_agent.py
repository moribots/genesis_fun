"""
Main script for training a PPO agent on the FrankaShelfEnv environment using RSL-RL.

This script handles:
- Initialization of the simulation environment (via the Genesis bridge).
- Configuration of the training environment and the RSL-RL PPO algorithm.
- Creation of the GenesisFrankaEnv vectorized environment.
- Instantiation and execution of the custom OnPolicyRunner, which manages the
  entire training loop.
"""
import os
import sys
import traceback
from dataclasses import asdict

import torch
import wandb

from sim_agnostic_core.training_core import TrainConfig, CustomOnPolicyRunner
# Import the concrete Genesis environment implementation
from genesis_bridge.genesis_env import GenesisFrankaEnv


def run_franka_training(cfg: TrainConfig) -> None:
    """
    Main orchestrator for the Franka agent training process.

    :param cfg: A TrainConfig object containing all configuration parameters.
    """
    env = None
    wandb_run = None

    try:
        # --- Handle W&B Login, config, and Initialization ---
        if cfg.runner.wandb:
            try:
                if wandb.api.api_key is None:
                    wandb.login()
                if cfg.runner.wandb_entity is None:
                    api = wandb.Api()
                    cfg.runner.wandb_entity = api.default_entity
                    if not cfg.runner.wandb_entity:
                        raise ValueError(
                            "Could not determine W&B default entity.")
                    print(
                        f"--- W&B entity not provided, using default: {cfg.runner.wandb_entity} ---")

                wandb_run = wandb.init(
                    project=cfg.runner.wandb_project,
                    entity=cfg.runner.wandb_entity,
                    group=cfg.runner.wandb_group or None,
                    name=cfg.runner.run_name or None,
                    tags=cfg.runner.wandb_tags,
                    config=asdict(cfg)
                )
            except Exception as e:
                print(f"--- W&B setup failed: {e}. Disabling W&B logging. ---")
                cfg.runner.wandb = False
                if wandb_run:
                    wandb_run.finish()

        # --- Create Environment ---
        # The environment-specific kwargs are now taken directly from the EnvConfig dataclass.
        # This makes the setup cleaner and less prone to errors.
        env_kwargs = asdict(cfg.env)
        # Add runner-specific device and render options
        env_kwargs['device'] = cfg.runner.device
        env_kwargs['render'] = False
        # RSL-RL expects max_steps_per_episode, not max_episode_length
        env_kwargs['max_steps_per_episode'] = env_kwargs.pop(
            'max_episode_length')

        # The key change: Instantiate the Genesis-specific environment
        env = GenesisFrankaEnv(**env_kwargs)
        print("GenesisFrankaEnv created successfully.")

        # --- Create Log Directory ---
        os.makedirs(cfg.runner.log_dir, exist_ok=True)

        # --- Instantiate and Run CustomOnPolicyRunner ---
        # The runner is now fully sim-agnostic and works with any env that
        # adheres to the BaseFrankaEnv interface.
        runner = CustomOnPolicyRunner(
            env, cfg, cfg.runner.log_dir, device=cfg.runner.device)
        print("CustomOnPolicyRunner instantiated. Starting training...")

        runner.learn(
            num_learning_iterations=cfg.runner.max_iterations,
            init_at_random_ep_len=True
        )
        print("Training finished.")

    except Exception as e:
        print(f"An unhandled exception occurred during training: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("Performing cleanup...")
        if env is not None:
            env.close()
            print("Environment and simulator closed.")
        if wandb_run:
            wandb_run.finish()
            print("W&B run finished.")
        print("Training script finished.")


if __name__ == '__main__':
    training_config = TrainConfig()
    run_franka_training(training_config)
