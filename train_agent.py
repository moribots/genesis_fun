"""
Main script for training a PPO agent on the FrankaShelfEnv environment.

This script handles:
- Initialization of the Genesis simulation environment.
- Configuration and setup of Weights & Biases (W&B) for experiment tracking.
- Creation and wrapping of the FrankaShelfEnv vectorized environment.
- Definition and instantiation of the PPO model from Stable Baselines3.
- Setup of custom callbacks for video recording, model checkpointing, evaluation,
  and W&B logging.
- The main training loop.
"""
import os
import sys
import getpass
from typing import Optional, List, Dict, Any

import traceback
import torch
import numpy as np
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

import genesis as gs  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.policies import ActorCriticPolicy
from wandb.integration.sb3 import WandbCallback as WandbCallbackSB3

from franka_rl_env import FrankaShelfEnv  # VecEnv under the hood
from collision_cnn import CustomCNNFeatureExtractor  # For mapping collision env


class VideoCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model checkpoint and recording/logging a video of the agent's performance.

    This callback extends the Stable Baselines3 CheckpointCallback to additionally
    trigger video recording and logging to Weights & Biases at specified intervals.

    :param save_freq: Save checkpoints every `save_freq` call of the callback.
    :param save_path: Path to the directory where checkpoints and videos will be saved.
    :param name_prefix: Prefix for checkpoint and video file names.
    :param video_log_freq_multiplier: Record and log a video every `video_log_freq_multiplier` checkpoints.
                                      For example, if `save_freq` is 1000 and `video_log_freq_multiplier` is 5,
                                      a video will be recorded every 5000 steps, aligned with checkpoint saves.
    :param video_length: Duration of the recorded video in environment steps.
    :param video_fps: Frames per second for the recorded video.
    :param verbose: Verbosity level (0 for no output, 1 for info messages).
    :param wandb_run: Active Weights & Biases run object for logging.
    """

    def __init__(self,
                 save_freq: int,
                 save_path: str,
                 name_prefix: str = "rl_model",
                 video_log_freq_multiplier: int = 1,
                 video_length: int = 200,
                 video_fps: int = 30,
                 verbose: int = 0,
                 wandb_run: Optional[WandbRun] = None):
        super().__init__(save_freq=save_freq, save_path=save_path,
                         name_prefix=name_prefix, verbose=verbose)
        self.video_length = video_length
        self.video_fps = video_fps
        self.wandb_run = wandb_run
        self.video_log_freq_multiplier = video_log_freq_multiplier
        self.checkpoint_count = 0  # Counter for checkpoints to control video logging frequency

    def _on_step(self) -> bool:
        """
        This method is called after each set of `n_steps` for the model.
        It handles checkpoint saving and, periodically, video recording and logging.

        :return: True if training should continue, False otherwise.
        """
        # First, execute the parent class's _on_step for checkpoint saving
        continue_training = super()._on_step()
        if not continue_training:
            return False

        # Check if a checkpoint was saved in this step by the parent
        if self.num_timesteps % self.save_freq == 0:
            self.checkpoint_count += 1

            # Determine if it's time to record a video
            if self.checkpoint_count % self.video_log_freq_multiplier == 0:
                if self.verbose > 0:
                    print(
                        f"Saving model checkpoint and recording video at timestep {self.num_timesteps}")

                video_filename = f"{self.name_prefix}_{self.num_timesteps}_video.mp4"
                # Ensure the save_path for videos exists (it's a subdirectory of the main save_path)
                video_save_dir = os.path.join(self.save_path, "videos")
                os.makedirs(video_save_dir, exist_ok=True)

                # Attempt to start video recording on the underlying FrankaShelfEnv
                # self.training_env is VecNormalize, self.training_env.venv is FrankaShelfEnv
                if not self.training_env.venv.start_video_recording(env_idx_to_focus=0):
                    if self.verbose > 0:
                        print("Failed to start video recording in environment.")
                    return True  # Continue training even if video recording setup fails

                if self.verbose > 0:
                    print(
                        f"Running dedicated episode for video recording (length {self.video_length})...")

                # Perform a rollout specifically for video recording
                # We need to handle the observation format for the model (potentially batched)
                obs_dict = self.training_env.reset()  # VecEnv reset
                # If model expects batch_size > 1, but we record one env, adjust obs
                # For simplicity, assuming predict can handle single env obs from VecEnv
                # or that the VecEnv correctly formats single env obs.

                ep_rewards_video = []
                temp_terminated_or_truncated = False
                current_obs_for_video = obs_dict  # Use the full batch from reset initially

                for _ in range(self.video_length):
                    if temp_terminated_or_truncated:
                        break  # Stop if episode ended prematurely

                    # Get actions for all environments in the training_env
                    all_actions, _ = self.model.predict(
                        current_obs_for_video, deterministic=True)

                    # Step all environments
                    next_obs_dict, rewards, dones, infos = self.training_env.step(
                        all_actions)
                    current_obs_for_video = next_obs_dict

                    # We are interested in the rewards and done status of the first environment (env_idx_to_focus=0)
                    ep_rewards_video.append(rewards[0])
                    if dones[0]:
                        temp_terminated_or_truncated = True

                # Stop recording and get the path to the saved video file
                video_file_path = self.training_env.venv.stop_video_recording(
                    save_dir=video_save_dir, filename=video_filename, fps=self.video_fps)

                # Log the video to W&B if a run is active and video was saved
                if video_file_path and self.wandb_run:
                    try:
                        log_payload = {
                            "media/training_progress_video": wandb.Video(video_file_path, fps=self.video_fps, format="mp4"),
                            # Log the mean reward of the episode recorded for the video
                            f"diagnostics/video_ep_mean_reward_env0_{self.num_timesteps}": safe_mean(ep_rewards_video) if ep_rewards_video else 0
                        }
                        self.wandb_run.log(
                            log_payload, step=self.num_timesteps)
                        if self.verbose > 0:
                            print(f"Logged video {video_file_path} to W&B.")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Error logging video to W&B: {e}")
        return True


def setup_wandb(project_name: Optional[str],
                entity: Optional[str],
                config: Dict[str, Any],
                monitor_gym: bool = True,
                save_code: bool = True) -> Optional[WandbRun]:
    """
    Sets up Weights & Biases for experiment tracking.

    Attempts to log in to W&B anonymously. If that fails and an API key
    is not already set, it prompts the user for an API key.
    Initializes a W&B run if a project name is provided.

    :param project_name: The name of the W&B project. If None, W&B is skipped.
    :param entity: The W&B entity (username or team name).
    :param config: A dictionary of hyperparameters and settings to log to W&B.
    :param monitor_gym: Whether to automatically log Gym environment stats (requires Monitor wrapper).
    :param save_code: Whether to save the main script to W&B.
    :return: The W&B run object if initialization is successful, otherwise None.
    """
    if not project_name:
        print("W&B project name not specified, skipping W&B logging.")
        return None

    run = None
    try:
        # Try to login, allowing anonymous usage if no API key is found
        wandb.login()
    except Exception:
        # This typically means WANDB_API_KEY is not set and interactive login is not available/failed
        print("Could not log in to W&B automatically. Ensure WANDB_API_KEY is set or run 'wandb login'.")
        try:
            # Fallback: prompt for API key if anonymous login didn't work out
            key = getpass.getpass(
                prompt="Enter your W&B API key (or press Enter to skip W&B): ")
            if key:
                os.environ["WANDB_API_KEY"] = key
                wandb.login()  # Attempt login with the provided key
            else:
                print("Skipping W&B logging as no API key was provided.")
                return None  # User chose to skip
        except Exception as e:
            print(
                f"Failed to log in to W&B with provided key: {e}. Skipping W&B.")
            return None  # Login with key failed

    # If login was successful (or anonymous allowed), initialize the run
    try:
        run = wandb.init(project=project_name,
                         entity=entity,
                         config=config,
                         sync_tensorboard=True,  # Essential for SB3 PPO metrics
                         monitor_gym=monitor_gym,
                         save_code=save_code)
        print(
            f"W&B run initialized: {run.url if run else 'None (failed init after login attempt?)'}")
    except Exception as e:
        print(f"Failed to initialize W&B run: {e}. Skipping W&B.")
        run = None

    return run


def create_env_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates the configuration dictionary for the FrankaShelfEnv.

    Extracts environment-specific parameters from the main training configuration.

    :param training_config: The main training configuration dictionary.
    :return: A dictionary of keyword arguments for FrankaShelfEnv.
    """
    return dict(
        render_mode=None,  # Typically None for training, can be 'human' for debugging
        num_envs=training_config["num_genesis_envs"],
        voxel_grid_dims=training_config["voxel_grid_dims_override"],
        voxel_grid_world_size=training_config["voxel_grid_world_size_override"],
        k_dist_reward=training_config.get('k_dist_reward'),
        k_time_penalty=training_config.get('k_time_penalty'),
        k_action_penalty=training_config.get('k_action_penalty'),
        k_joint_limit_penalty=training_config.get(
            'k_joint_limit_penalty'),
        k_collision_penalty=training_config.get('k_collision_penalty'),
        k_accel_penalty=training_config.get('k_accel_penalty'),
        success_reward_val=training_config.get('success_reward_val'),
        success_threshold_val=training_config.get(
            'success_threshold_val'),
        max_steps_per_episode=training_config.get(
            'max_steps_per_episode'),
        video_camera_pos=tuple(training_config.get(
            'video_camera_pos_override')),
        video_camera_lookat=tuple(training_config.get(
            'video_camera_lookat_override')),
        video_camera_fov=training_config.get('video_camera_fov_override'),
        video_res=tuple(training_config.get('video_res_override'))
    )


def create_training_env(num_genesis_envs: int,
                        seed: int,
                        env_kwargs: Dict[str, Any],
                        gamma: float) -> VecNormalize:
    """
    Creates and wraps the FrankaShelfEnv for training.

    Instantiates FrankaShelfEnv, seeds it, and wraps it with VecNormalize.

    :param num_genesis_envs: Number of parallel Genesis environments.
    :param seed: Random seed for the environment.
    :param env_kwargs: Keyword arguments for FrankaShelfEnv.
    :param gamma: Discount factor for rewards, used by VecNormalize.
    :return: The wrapped VecNormalize training environment.
    """
    try:
        # FrankaShelfEnv is expected to be a VecEnv-compatible class
        env = FrankaShelfEnv(**env_kwargs)
        env.seed(seed)  # Seed the custom VecEnv
    except Exception as e:
        print(f"Error initializing FrankaShelfEnv: {e}")
        traceback.print_exc()
        sys.exit(1)  # Critical failure

    # Basic check for VecEnv interface compatibility
    if not (hasattr(env, 'num_envs') and hasattr(env, 'observation_space') and
            hasattr(env, 'action_space') and callable(getattr(env, 'reset', None)) and
            callable(getattr(env, 'step_async', None)) and callable(getattr(env, 'step_wait', None))):
        print("ERROR: FrankaShelfEnv does not seem to have the full VecEnv interface.")
        sys.exit(1)  # Critical failure

    if env.num_envs != num_genesis_envs:
        print(
            f"Warning: FrankaShelfEnv.num_envs ({env.num_envs}) != requested ({num_genesis_envs}). "
            f"Using {env.num_envs}.")

    # Wrap the environment with VecNormalize for observation and reward normalization
    # Note: If FrankaShelfEnv is already a VecEnv, it's directly wrapped.
    # If it were a single Gym env, it would typically be wrapped by DummyVecEnv first.
    normalized_env = VecNormalize(
        env, norm_obs=True, norm_reward=True, gamma=gamma)
    print(
        f"FrankaShelfEnv (VecEnv with {normalized_env.num_envs} internal envs) wrapped with VecNormalize.")
    return normalized_env


def create_ppo_model(config: Dict[str, Any],
                     env: VecNormalize,
                     policy_kwargs: Optional[Dict[str, Any]],
                     log_dir: str) -> PPO:
    """
    Creates the PPO (Proximal Policy Optimization) agent.

    :param config: The main training configuration dictionary.
    :param env: The training environment (VecNormalize instance).
    :param policy_kwargs: Keyword arguments for the policy network (e.g., feature extractor).
    :param log_dir: Directory for saving TensorBoard logs.
    :return: The initialized PPO model.
    """
    # PPO model instantiation
    model = PPO(
        policy=config["policy_type"],
        env=env,
        verbose=1,
        tensorboard_log=log_dir,  # Always provide log_dir for SB3 internal logging
        seed=config["seed"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_steps=config["n_steps_ppo"],  # Rollout buffer size per environment
        batch_size=config["batch_size_ppo"],
        n_epochs=config["n_epochs_ppo"],
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=policy_kwargs
    )
    print(
        f"PPO model created. Rollout buffer size: {model.n_steps} per env * "
        f"{model.n_envs} envs = {model.n_steps * model.n_envs} total samples per update.")
    return model


def create_callbacks(config: Dict[str, Any],
                     model: PPO,  # model.n_steps and model.n_envs are used
                     eval_env: VecNormalize,  # The environment for evaluation
                     log_dir: str,
                     model_save_path_base: str,  # Base path for model, not specific checkpoint
                     wandb_run_instance: Optional[WandbRun]) -> List[BaseCallback]:
    """
    Creates a list of callbacks for the training process.

    Includes VideoCheckpointCallback, EvalCallback, and WandbCallback.

    :param config: The main training configuration dictionary.
    :param model: The PPO model (used to determine rollout buffer size for frequency calculations).
    :param eval_env: The environment to use for evaluation.
    :param log_dir: Directory for logs and checkpoints.
    :param model_save_path_base: Base name/path for saving model checkpoints (e.g., "ppo_franka_shelf").
                                 Actual checkpoints will have timesteps appended.
    :param wandb_run_instance: Active Weights & Biases run object.
    :return: A list of callback instances.
    """
    callbacks = []

    # Calculate frequencies in terms of total environment steps
    samples_per_ppo_rollout = model.n_steps * model.n_envs
    checkpoint_save_freq_steps = samples_per_ppo_rollout * \
        config["checkpoint_save_freq_rollouts"]
    eval_freq_steps = samples_per_ppo_rollout * config["eval_freq_rollouts"]

    print(
        f"Checkpoint save frequency: every {checkpoint_save_freq_steps} total env steps.")
    print(f"Evaluation frequency: every {eval_freq_steps} total env steps.")

    # Video recording and model checkpointing callback
    video_checkpoint_cb = VideoCheckpointCallback(
        save_freq=checkpoint_save_freq_steps,
        save_path=log_dir,  # Checkpoints and videos subfolder will be created here
        name_prefix=f"{os.path.basename(model_save_path_base)}_ckpt",
        video_log_freq_multiplier=config["video_log_freq_multiplier"],
        video_length=config["video_length"],
        video_fps=config["video_fps"],
        wandb_run=wandb_run_instance,
        verbose=1
    )
    callbacks.append(video_checkpoint_cb)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env=eval_env,  # Use the same normalized training env for evaluation
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq_steps,
        # Number of episodes to run for evaluation, scaled by number of envs
        n_eval_episodes=max(5 * eval_env.num_envs // 2,
                            eval_env.num_envs if eval_env.num_envs > 0 else 5),
        deterministic=True,
        render=False,  # No rendering during evaluation for speed
        verbose=1
    )
    callbacks.append(eval_callback)

    # W&B callback for logging metrics, model artifacts, etc.
    if wandb_run_instance:
        wandb_sb3_callback = WandbCallbackSB3(
            model_save_path=os.path.join(
                log_dir, f"wandb_models/{wandb_run_instance.id}"),
            model_save_freq=samples_per_ppo_rollout *
            config.get("wandb_model_save_freq_rollouts"),
            log="all",  # Log all SB3 metrics (gradients, histograms, etc.)
            verbose=2
        )
        callbacks.append(wandb_sb3_callback)

    return callbacks


def train_and_save_model(model: PPO,
                         config: Dict[str, Any],
                         callbacks: List[BaseCallback],
                         model_save_path_base: str,  # Base path for model, e.g., ./logs/model_name
                         env: VecNormalize,
                         # Directory where logs (and final model) are saved
                         log_dir: str,
                         wandb_run_instance: Optional[WandbRun]) -> None:
    """
    Runs the main training loop and handles final model saving and W&B artifact logging.

    :param model: The PPO model to train.
    :param config: The main training configuration dictionary.
    :param callbacks: List of callbacks to use during training.
    :param model_save_path_base: Base name/path for the model. The final model will be
                                 saved as `{model_save_path_base}_final.zip`.
    :param env: The training environment (VecNormalize instance), used for saving normalization stats.
    :param log_dir: The directory where logs are stored. The final model and stats
                    will also be saved relative to this or `model_save_path_base`.
    :param wandb_run_instance: Active Weights & Biases run object for artifact logging.
    """
    print(
        f"Starting training for {config['total_timesteps']} total environment steps...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True  # Show a progress bar during training
        )
    except Exception as e:
        print(f"Error during model training: {e}")

        traceback.print_exc()
    finally:
        # Determine save directory and final model name
        # If model_save_path_base is a full path like "dir/model_name", use its dirname.
        # If it's just "model_name", save in log_dir.
        if os.path.dirname(model_save_path_base):
            save_dir = os.path.dirname(model_save_path_base)
            final_model_name_stem = os.path.basename(model_save_path_base)
        else:
            save_dir = log_dir
            final_model_name_stem = model_save_path_base

        os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

        full_final_model_path = os.path.join(
            save_dir, f"{final_model_name_stem}_final.zip")
        model.save(full_final_model_path)
        print(f"Final model saved to {full_final_model_path}")

        # Save VecNormalize statistics
        vec_normalize_stats_path = os.path.join(
            save_dir, f"{final_model_name_stem}_vecnormalize.pkl")
        if isinstance(env, VecNormalize):  # Should always be true with current setup
            env.save(vec_normalize_stats_path)
            print(f"VecNormalize stats saved to {vec_normalize_stats_path}")

        # Log final model and stats as W&B artifacts if a run is active
        if wandb_run_instance:
            print("Logging final model and VecNormalize stats to W&B as artifacts...")
            try:
                trained_model_artifact = wandb.Artifact(
                    name=f'{final_model_name_stem}_final_model', type='model')
                trained_model_artifact.add_file(full_final_model_path)
                wandb_run_instance.log_artifact(trained_model_artifact)

                if isinstance(env, VecNormalize) and os.path.exists(vec_normalize_stats_path):
                    vec_stats_artifact = wandb.Artifact(
                        name=f'{final_model_name_stem}_vecnormalize_stats', type='dataset')
                    vec_stats_artifact.add_file(vec_normalize_stats_path)
                    wandb_run_instance.log_artifact(vec_stats_artifact)
                print("Artifacts logged to W&B.")
            except Exception as e:
                print(f"Error logging artifacts to W&B: {e}")


def cleanup_training(env: VecNormalize, wandb_run_instance: Optional[WandbRun]) -> None:
    """
    Performs cleanup operations after training.

    Closes the training environment, finishes the W&B run, and shuts down Genesis.

    :param env: The training environment.
    :param wandb_run_instance: Active Weights & Biases run object.
    """
    if env:
        env.close()
        print("Training environment closed.")

    if wandb_run_instance:
        wandb_run_instance.finish()
        print("W&B run finished.")
    print("Training script finished.")


def run_franka_training(training_config: Dict[str, Any]) -> None:
    """
    Main orchestrator for the Franka agent training process.

    :param training_config: A dictionary containing all parameters for the training run.
    """
    # --- Initialization ---
    gs.init(backend=gs.gpu)
    set_random_seed(training_config["seed"])

    # --- W&B Setup ---
    wandb_run = setup_wandb(
        project_name=training_config.get("wandb_project_name"),
        entity=training_config.get("wandb_entity"),
        config=training_config
    )

    # --- Directory Setup ---
    log_dir = training_config["log_dir"]
    # Base path for model, e.g. ./logs/model_name
    model_save_path_base = training_config["model_save_path"]
    os.makedirs(log_dir, exist_ok=True)
    # If model_save_path includes a directory
    if os.path.dirname(model_save_path_base):
        os.makedirs(os.path.dirname(model_save_path_base), exist_ok=True)

    # --- Environment Setup ---
    env_specific_config = create_env_config(training_config)
    if wandb_run:  # Log environment specific config to W&B if not already part of main config
        wandb_env_config_to_log = {
            k: v for k, v in env_specific_config.items() if k not in training_config}
        if wandb_env_config_to_log:
            wandb.config.update(wandb_env_config_to_log, allow_val_change=True)

    train_env = create_training_env(
        num_genesis_envs=training_config["num_genesis_envs"],
        seed=training_config["seed"],
        env_kwargs=env_specific_config,
        gamma=training_config["gamma"]
    )

    # --- Policy and Model Setup ---
    policy_kwargs = None
    if training_config["use_cnn"]:
        policy_kwargs = dict(
            features_extractor_class=CustomCNNFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=training_config.get("cnn_features_dim"))
        )

    ppo_model = create_ppo_model(
        config=training_config,
        env=train_env,
        policy_kwargs=policy_kwargs,
        log_dir=log_dir
    )

    # --- Callbacks Setup ---
    # For EvalCallback, it's common to use the same training environment instance,
    # as VecNormalize handles its own normalization statistics for evaluation.
    training_callbacks = create_callbacks(
        config=training_config,
        model=ppo_model,
        eval_env=train_env,  # Using the normalized training env for evaluation
        log_dir=log_dir,
        model_save_path_base=model_save_path_base,
        wandb_run_instance=wandb_run
    )

    # --- Training and Saving ---
    train_and_save_model(
        model=ppo_model,
        config=training_config,
        callbacks=training_callbacks,
        model_save_path_base=model_save_path_base,
        env=train_env,
        log_dir=log_dir,
        wandb_run_instance=wandb_run
    )

    # --- Cleanup ---
    cleanup_training(train_env, wandb_run)


if __name__ == '__main__':
    # Configuration parameters for the training run.
    # This single dictionary holds all settings.
    config_params = {
        "policy_type": "MultiInputPolicy",
        "seed": 42,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps_ppo": 32,
        "batch_size_ppo": 2048,        # Larger mini-batches to leverage gpu
        "n_epochs_ppo": 10,
        "learning_rate": 6e-4,         # Increased from 3e-4 to mesh with batch size update
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_cnn": True,
        "cnn_features_dim": 256,
        "total_timesteps": 15_000_000,
        "num_genesis_envs": 1024,
        "voxel_grid_world_size_override": (0.8, 0.8, 0.5),
        "voxel_grid_dims_override": (32, 48, 48),
        "log_dir": "./training_logs/ppo_franka_shelf_refactored/",
        "model_save_path": "./training_logs/ppo_franka_shelf_refactored/model_refactored",
        "wandb_project_name": "FrankaShelfPPO-Refactored",
        "wandb_entity": None,
        "wandb_model_save_freq_rollouts": 20,
        "checkpoint_save_freq_rollouts": 3,
        "eval_freq_rollouts": 20,
        "video_log_freq_multiplier": 2,
        "video_length": 750,
        "video_fps": 30,
        "k_dist_reward": 1.5,
        "k_time_penalty": 0.02,
        "k_action_penalty": 0.005,
        "k_joint_limit_penalty": 15.0,
        "k_collision_penalty": 200.0,
        "k_accel_penalty": 0.005,
        "success_reward_val": 350.0,
        "success_threshold_val": 0.05,
        "max_steps_per_episode": 750,
        'video_camera_pos_override': (1.8, -1.8, 2.0),
        'video_camera_lookat_override': (0.3, 0.0, 0.5),
        'video_camera_fov_override': 45,
        'video_res_override': (960, 640)
    }

    print("--- Franka Shelf PPO Training (Refactored Script with Single Config) ---")
    for key, val in config_params.items():
        print(f"{key}: {val}")
    print("-----------------------------------------------------------------------")

    run_franka_training(training_config=config_params)
