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
import numbers

import traceback
import torch
import numpy as np
import wandb
from wandb.sdk.wandb_run import Run as WandbRun

import genesis as gs  # type: ignore

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.policies import ActorCriticPolicy  # Used by PPO
from wandb.integration.sb3 import WandbCallback as WandbCallbackSB3

# Environment with shelf parameter observation
from franka_rl_env import FrankaShelfEnv


class WandbDetailedMetricsCallback(BaseCallback):
    """
    A custom callback to log detailed metrics from the environment's info dictionary
    directly to Weights & Biases at the end of each rollout.
    It calculates the mean of specified scalar metrics collected across all steps
    and all environments within the rollout.
    """

    def __init__(self, wandb_run_instance: Optional[WandbRun], verbose=0):
        super().__init__(verbose)
        self.wandb_run = wandb_run_instance
        # Define the keys from the info dictionary that we want to log
        self.metrics_to_track = [
            "diagnostics/distance_to_target",
            "rewards/distance", "rewards/time_penalty",
            "rewards/action_penalty", "rewards/joint_limit_penalty",
            "rewards/collision_penalty", "rewards/success", "rewards/accel_penalty",
            "diagnostics/ee_pos_x", "diagnostics/ee_pos_y", "diagnostics/ee_pos_z",
            "diagnostics/target_pos_x", "diagnostics/target_pos_y", "diagnostics/target_pos_z",
            "diagnostics/mean_qpos_j1-3", "diagnostics/mean_qvel_j1-3",
            "is_success"  # Also log success rate within the rollout
        ]
        self.rollout_data = {key: [] for key in self.metrics_to_track}

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(
                f"--- WandbDetailedMetricsCallback: Training started. Logging to W&B run: {self.wandb_run is not None} ---")
            print(
                f"--- WandbDetailedMetricsCallback: Tracking metrics: {self.metrics_to_track} ---")

    def _on_rollout_start(self) -> None:
        """
        This method is called before a new rollout starts.
        Clear accumulated metrics.
        """
        for key in self.metrics_to_track:
            self.rollout_data[key] = []

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        `self.locals["infos"]` is a list of info dicts for each environment at the current step.
        """
        infos_list = self.locals.get("infos", [])
        for env_info in infos_list:
            if env_info is None:
                continue
            for key in self.metrics_to_track:
                if key in env_info:
                    value = env_info[key]
                    # Ensure value is scalar (int, float, or bool which converts to 0/1)
                    if isinstance(value, (int, float, bool, np.bool_)):
                        self.rollout_data[key].append(float(value))
        return True

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout.
        Calculate means and log to W&B.
        """
        metrics_to_log_wandb = {}
        for key, values_list in self.rollout_data.items():
            if values_list:  # If we collected any data for this key
                mean_value = float(np.mean(values_list))
                # Log to SB3 logger (WandbCallbackSB3 might pick this up too if it logs all from logger)
                self.logger.record(f"custom_rollout_stats/{key}", mean_value)
                # Prepare for direct W&B logging
                metrics_to_log_wandb[f"custom_rollout_stats/{key}"] = mean_value

        if metrics_to_log_wandb and self.wandb_run is not None:
            # MOD: Removed step=self.num_timesteps to avoid conflict with tensorboard syncing
            self.wandb_run.log(metrics_to_log_wandb)
            if self.verbose > 0:
                print(
                    f"--- WandbDetailedMetricsCallback: Logged {len(metrics_to_log_wandb)} custom metrics to W&B at total steps {self.num_timesteps} ---")

        # Clear for the next rollout (already done in _on_rollout_start, but good for safety)
        for key in self.metrics_to_track:
            self.rollout_data[key] = []


class VideoCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model checkpoint and recording/logging a video.
    It extends the CheckpointCallback to also trigger video recording
    at specified intervals (multiples of checkpoint frequency).
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
        self.video_log_freq_multiplier = max(1, video_log_freq_multiplier)
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training:
            return False

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self.checkpoint_count += 1

            if self.checkpoint_count % self.video_log_freq_multiplier == 0:
                if self.verbose > 0:
                    print(
                        f"Saving model checkpoint and recording video at timestep {self.num_timesteps}")

                video_filename = f"{self.name_prefix}_{self.num_timesteps}_video.mp4"
                video_save_dir = os.path.join(self.save_path, "videos")
                os.makedirs(video_save_dir, exist_ok=True)

                actual_env_for_video = self.training_env.venv if hasattr(
                    self.training_env, 'venv') else self.training_env

                if not hasattr(actual_env_for_video, 'start_video_recording') or \
                   not hasattr(actual_env_for_video, 'stop_video_recording'):
                    if self.verbose > 0:
                        print("Video recording methods not found on the environment.")
                    return True

                if not actual_env_for_video.start_video_recording(env_idx_to_focus=0):
                    if self.verbose > 0:
                        print("Failed to start video recording in environment.")
                    return True

                if self.verbose > 0:
                    print(
                        f"Running dedicated episode for video recording (length {self.video_length})...")

                obs_for_video_predict = self.training_env.buf_obs if hasattr(
                    self.training_env, 'buf_obs') else self.training_env.reset()

                ep_rewards_video = []
                dones_video = np.array([False] * self.training_env.num_envs)

                for _ in range(self.video_length):
                    if dones_video[0]:
                        break
                    all_actions, _ = self.model.predict(
                        obs_for_video_predict, deterministic=True)
                    next_obs_dict, rewards, dones_step, infos = self.training_env.step(
                        all_actions)
                    obs_for_video_predict = next_obs_dict
                    ep_rewards_video.append(rewards[0])
                    dones_video = dones_step

                video_file_path = actual_env_for_video.stop_video_recording(
                    save_dir=video_save_dir, filename=video_filename, fps=self.video_fps)

                if video_file_path and self.wandb_run:
                    try:
                        log_payload = {
                            f"media/training_video_step": wandb.Video(video_file_path, fps=self.video_fps, format="mp4"),
                            f"diagnostics/video_ep_mean_reward_env0_step": safe_mean(ep_rewards_video) if ep_rewards_video else 0
                        }
                        self.wandb_run.log(
                            log_payload, step=self.num_timesteps)
                        if self.verbose > 0:
                            print(f"Logged video {video_file_path} to W&B.")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Error logging video to W&B: {e}")
        return True


def setup_wandb(project_name: Optional[str], entity: Optional[str], config: Dict[str, Any],
                monitor_gym: bool = True, save_code: bool = True) -> Optional[WandbRun]:
    """ 
    Sets up Weights & Biases for experiment tracking.
    Handles login and initialization of a W&B run.
    """
    if not project_name:
        print("W&B project name not provided. Skipping W&B setup.")
        return None

    run: Optional[WandbRun] = None
    try:
        wandb.login()
    except Exception:
        try:
            print("W&B login failed. Please enter your W&B API key.")
            key = getpass.getpass(
                prompt="Enter your W&B API key (or press Enter to skip W&B): ")
            if key:
                os.environ["WANDB_API_KEY"] = key
                wandb.login()
            else:
                print("No W&B API key entered. Skipping W&B.")
                return None
        except Exception as e:
            print(
                f"Error during W&B login with manual key: {e}. Skipping W&B.")
            return None

    try:
        run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            sync_tensorboard=True,
            monitor_gym=monitor_gym,
            save_code=save_code
        )
        print(
            f"W&B run initialized successfully. Run URL: {run.url if run else 'N/A'}")
    except Exception as e:
        print(f"Error initializing W&B run: {e}. Proceeding without W&B.")
        run = None
    return run


def create_env_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """ 
    Creates the configuration dictionary specifically for initializing FrankaShelfEnv.
    Extracts relevant parameters from the main training_config.
    """
    return dict(
        render_mode=None,
        num_envs=training_config["num_genesis_envs"],
        workspace_bounds_xyz=tuple(training_config.get(
            'workspace_bounds_xyz_override', ((-1.5, 1.5), (-1.5, 1.5), (0.0, 3.0)))),
        k_dist_reward=training_config.get('k_dist_reward'),
        k_time_penalty=training_config.get('k_time_penalty'),
        k_action_penalty=training_config.get('k_action_penalty'),
        k_joint_limit_penalty=training_config.get('k_joint_limit_penalty'),
        k_collision_penalty=training_config.get('k_collision_penalty'),
        k_accel_penalty=training_config.get('k_accel_penalty'),
        success_reward_val=training_config.get('success_reward_val'),
        success_threshold_val=training_config.get('success_threshold_val'),
        max_steps_per_episode=training_config.get('max_steps_per_episode'),
        video_camera_pos=tuple(training_config.get(
            'video_camera_pos_override')),
        video_camera_lookat=tuple(training_config.get(
            'video_camera_lookat_override')),
        video_camera_fov=training_config.get('video_camera_fov_override'),
        video_res=tuple(training_config.get('video_res_override')),
        include_shelf=training_config.get('include_shelf', True),
        randomize_shelf_config=training_config.get(
            'randomize_shelf_config', True)
    )


def create_training_env(num_genesis_envs: int, seed: int, env_kwargs: Dict[str, Any], gamma: float) -> VecNormalize:
    """ 
    Creates the FrankaShelfEnv, seeds it, and wraps it with VecNormalize.
    """
    try:
        env = FrankaShelfEnv(**env_kwargs)
        env.seed(seed)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to create FrankaShelfEnv: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not (hasattr(env, 'num_envs') and hasattr(env, 'observation_space') and hasattr(env, 'action_space') and
            callable(getattr(env, 'reset', None)) and callable(getattr(env, 'step_async', None)) and
            callable(getattr(env, 'step_wait', None))):
        print("CRITICAL ERROR: Created environment does not conform to VecEnv interface.")
        sys.exit(1)

    if env.num_envs != num_genesis_envs:
        print(
            f"Warning: FrankaShelfEnv num_envs ({env.num_envs}) differs from requested num_genesis_envs ({num_genesis_envs}).")

    normalized_env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        gamma=gamma
    )
    return normalized_env


def create_ppo_model(config: Dict[str, Any], env: VecNormalize,
                     policy_kwargs: Optional[Dict[str, Any]], log_dir: str,
                     wandb_run: Optional[WandbRun] = None) -> PPO:
    """ 
    Creates and returns a PPO agent from Stable Baselines3.
    """
    model = PPO(
        policy=config["policy_type"],
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        seed=config["seed"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        n_steps=config["n_steps_ppo"],
        batch_size=config["batch_size_ppo"],
        n_epochs=config["n_epochs_ppo"],
        learning_rate=config["learning_rate"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs=policy_kwargs
    )
    return model


def create_callbacks(config: Dict[str, Any], model: PPO, eval_env: VecNormalize,
                     log_dir: str, model_save_path_base: str,
                     wandb_run_instance: Optional[WandbRun]) -> List[BaseCallback]:
    """ 
    Creates a list of callbacks for training, including checkpointing, evaluation, and W&B logging.
    """
    callbacks = []

    samples_per_ppo_rollout = model.n_steps * model.n_envs

    checkpoint_save_freq_steps = samples_per_ppo_rollout * \
        config["checkpoint_save_freq_rollouts"]
    eval_freq_steps = samples_per_ppo_rollout * config["eval_freq_rollouts"]

    video_checkpoint_cb = VideoCheckpointCallback(
        save_freq=checkpoint_save_freq_steps,
        save_path=log_dir,
        name_prefix=f"{os.path.basename(model_save_path_base)}_ckpt",
        video_log_freq_multiplier=config["video_log_freq_multiplier"],
        video_length=config["video_length"],
        video_fps=config["video_fps"],
        wandb_run=wandb_run_instance,
        verbose=1
    )
    callbacks.append(video_checkpoint_cb)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq_steps,
        n_eval_episodes=max(5 * eval_env.num_envs // 2,
                            eval_env.num_envs if eval_env.num_envs > 0 else 5),
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    # MOD: Add the new WandbDetailedMetricsCallback
    if wandb_run_instance:
        detailed_metrics_cb = WandbDetailedMetricsCallback(
            wandb_run_instance=wandb_run_instance, verbose=1)
        callbacks.append(detailed_metrics_cb)

        # The original WandbCallbackSB3 can still be used for its other logging features
        # (model checkpoints to W&B, default SB3 logger metrics, etc.)
        wandb_sb3_callback = WandbCallbackSB3(
            model_save_path=os.path.join(
                log_dir, f"wandb_models/{wandb_run_instance.id}"),
            model_save_freq=samples_per_ppo_rollout *
            config.get("wandb_model_save_freq_rollouts", 0),
            gradient_save_freq=samples_per_ppo_rollout *
            config.get("wandb_model_save_freq_rollouts", 0),
            log="all",  # This will continue to log SB3 default logger content
            verbose=2
        )
        callbacks.append(wandb_sb3_callback)

    return callbacks


def train_and_save_model(model: PPO, config: Dict[str, Any], callbacks: List[BaseCallback],
                         model_save_path_base: str, env: VecNormalize, log_dir: str,
                         wandb_run_instance: Optional[WandbRun]) -> None:
    """ 
    Runs the main training loop and handles final model and VecNormalize stats saving.
    Also logs artifacts to W&B if enabled.
    """
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
    finally:
        save_dir = os.path.dirname(model_save_path_base) if os.path.dirname(
            model_save_path_base) else log_dir
        final_model_name_stem = os.path.basename(model_save_path_base)
        os.makedirs(save_dir, exist_ok=True)

        full_final_model_path = os.path.join(
            save_dir, f"{final_model_name_stem}_final.zip")
        model.save(full_final_model_path)
        print(f"Final model saved to {full_final_model_path}")

        vec_normalize_stats_path = os.path.join(
            save_dir, f"{final_model_name_stem}_vecnormalize.pkl")
        if isinstance(env, VecNormalize):
            env.save(vec_normalize_stats_path)
            print(f"VecNormalize stats saved to {vec_normalize_stats_path}")

        if wandb_run_instance:
            try:
                trained_model_artifact = wandb.Artifact(
                    name=f'{final_model_name_stem}_final_model', type='model')
                trained_model_artifact.add_file(full_final_model_path)
                wandb_run_instance.log_artifact(trained_model_artifact)
                print(
                    f"Logged final model to W&B: {trained_model_artifact.name}")

                if isinstance(env, VecNormalize) and os.path.exists(vec_normalize_stats_path):
                    vec_stats_artifact = wandb.Artifact(
                        name=f'{final_model_name_stem}_vecnormalize_stats', type='dataset')
                    vec_stats_artifact.add_file(vec_normalize_stats_path)
                    wandb_run_instance.log_artifact(vec_stats_artifact)
                    print(
                        f"Logged VecNormalize stats to W&B: {vec_stats_artifact.name}")
            except Exception as e:
                print(f"Error logging final artifacts to W&B: {e}")


def cleanup_training(env: Optional[VecNormalize], wandb_run_instance: Optional[WandbRun]) -> None:
    """ 
    Performs cleanup operations after training, such as closing the environment
    and finishing the W&B run.
    """
    if env:
        try:
            env.close()
            print("Training environment closed.")
        except Exception as e:
            print(f"Error closing environment: {e}")

    if wandb_run_instance:
        try:
            wandb_run_instance.finish()
            print("W&B run finished.")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")


def run_franka_training(training_config: Dict[str, Any]) -> None:
    """ 
    Main orchestrator for the Franka agent training process.
    Initializes Genesis, sets up W&B, creates environment and agent,
    manages training, and handles cleanup.
    """
    gs_initialized_locally = False
    train_env_instance: Optional[VecNormalize] = None
    wandb_run: Optional[WandbRun] = None

    try:
        if not (hasattr(gs, '_is_initialized') and gs._is_initialized):
            backend_to_use = gs.gpu if torch.cuda.is_available() else gs.cpu
            print(
                f"Initializing Genesis with backend: {'GPU' if backend_to_use == gs.gpu else 'CPU'}")
            gs.init(backend=backend_to_use)
            gs_initialized_locally = True
            print("Genesis initialized successfully.")

        set_random_seed(training_config["seed"])
        print(f"Random seed set to: {training_config['seed']}")

        wandb_run = setup_wandb(
            project_name=training_config.get("wandb_project_name"),
            entity=training_config.get("wandb_entity"),
            config=training_config
        )

        log_dir = training_config["log_dir"]
        model_save_path_base = training_config["model_save_path"]
        os.makedirs(log_dir, exist_ok=True)
        if os.path.dirname(model_save_path_base):
            os.makedirs(os.path.dirname(model_save_path_base), exist_ok=True)
        print(f"Log directory: {log_dir}")
        print(f"Base model save path: {model_save_path_base}")

        env_specific_config = create_env_config(training_config)
        if wandb_run:
            wandb_env_config_to_log = {
                f"env_{k}": v for k, v in env_specific_config.items() if k not in training_config}
            if wandb_env_config_to_log:
                wandb_run.config.update(wandb_env_config_to_log,
                                        allow_val_change=True)

        train_env_instance = create_training_env(
            num_genesis_envs=training_config["num_genesis_envs"],
            seed=training_config["seed"],
            env_kwargs=env_specific_config,
            gamma=training_config["gamma"]
        )
        print("Training environment created and normalized.")

        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        print(f"Using policy_kwargs: {policy_kwargs}")

        ppo_model = create_ppo_model(
            config=training_config,
            env=train_env_instance,
            policy_kwargs=policy_kwargs,
            log_dir=log_dir,
            wandb_run=wandb_run
        )
        print(
            f"PPO model created with policy: {training_config['policy_type']}")

        training_callbacks = create_callbacks(
            config=training_config, model=ppo_model, eval_env=train_env_instance,
            log_dir=log_dir, model_save_path_base=model_save_path_base,
            wandb_run_instance=wandb_run
        )
        print(f"Created {len(training_callbacks)} callbacks for training.")

        print("Starting model training...")
        train_and_save_model(
            model=ppo_model, config=training_config, callbacks=training_callbacks,
            model_save_path_base=model_save_path_base, env=train_env_instance,
            log_dir=log_dir, wandb_run_instance=wandb_run
        )
        print("Model training finished.")

    except Exception as e:
        print(f"An unhandled exception occurred during training: {e}")
        traceback.print_exc()
    finally:
        print("Performing cleanup...")
        cleanup_training(train_env_instance, wandb_run)

        if gs_initialized_locally and hasattr(gs, 'shutdown') and callable(gs.shutdown):
            try:
                gs.shutdown()
                print("Genesis shut down successfully.")
            except Exception as e:
                print(f"Error during Genesis shutdown: {e}")
        print("Training script finished.")


if __name__ == '__main__':
    config_params = {
        "policy_type": "MultiInputPolicy",
        "seed": 42,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps_ppo": 128,
        "batch_size_ppo": 2048,
        "n_epochs_ppo": 5,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "vf_coef": 1.0,  # value function coefficient
        "max_grad_norm": 0.5,
        "total_timesteps": 100_000_000,
        "num_genesis_envs": 1024,

        "workspace_bounds_xyz_override": ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.5)),
        "include_shelf": False,
        "randomize_shelf_config": False,

        "log_dir": "./training_logs/ppo_franka_shelf_params_obs/",
        "model_save_path": "./training_logs/ppo_franka_shelf_params_obs/model_shelf_params",
        "wandb_project_name": "FrankaPPO-ShelfParamsObs",
        "wandb_entity": None,
        "wandb_model_save_freq_rollouts": 5,

        "checkpoint_save_freq_rollouts": 5,
        "eval_freq_rollouts": 5,
        "video_log_freq_multiplier": 3,
        "video_length": 300,
        "video_fps": 30,

        "k_dist_reward": 20.0,
        "k_time_penalty": 0.02,
        "k_action_penalty": 0.005,
        "k_joint_limit_penalty": 15.0,
        "k_collision_penalty": 50.0,
        "k_accel_penalty": 0.005,
        "success_reward_val": 350.0,
        "success_threshold_val": 0.05,
        "max_steps_per_episode": 500,        # MODIFIED from 300

        'video_camera_pos_override': (1.8, -1.8, 2.0),
        'video_camera_lookat_override': (0.3, 0.0, 0.5),
        'video_camera_fov_override': 45,
        'video_res_override': (960, 640)
    }

    run_franka_training(training_config=config_params)
