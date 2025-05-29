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
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.policies import ActorCriticPolicy  # Used by PPO
from wandb.integration.sb3 import WandbCallback as WandbCallbackSB3

# Environment with shelf parameter observation
from franka_rl_env import FrankaShelfEnv


class VideoCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model checkpoint and recording/logging a video.
    (Docstring and implementation details remain the same as previous correct versions)
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
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training:
            return False
        if self.num_timesteps % self.save_freq == 0:  # Checkpoint saved, num_timesteps is a multiple of save_freq
            self.checkpoint_count += 1
            if self.checkpoint_count % self.video_log_freq_multiplier == 0:  # Time to log video
                if self.verbose > 0:
                    print(
                        f"Saving model checkpoint and recording video at timestep {self.num_timesteps}")
                video_filename = f"{self.name_prefix}_{self.num_timesteps}_video.mp4"
                video_save_dir = os.path.join(self.save_path, "videos")
                os.makedirs(video_save_dir, exist_ok=True)

                # Access the underlying FrankaShelfEnv.
                # self.training_env is likely VecNormalize, so .venv gets the FrankaShelfEnv
                actual_env_for_video = self.training_env.venv if hasattr(
                    self.training_env, 'venv') else self.training_env

                if not hasattr(actual_env_for_video, 'start_video_recording') or \
                   not hasattr(actual_env_for_video, 'stop_video_recording'):
                    if self.verbose > 0:
                        print("Video recording methods not found on the environment.")
                    return True  # Continue training

                if not actual_env_for_video.start_video_recording(env_idx_to_focus=0):
                    if self.verbose > 0:
                        print("Failed to start video recording in environment.")
                    return True  # Continue training

                if self.verbose > 0:
                    print(
                        f"Running dedicated episode for video recording (length {self.video_length})...")

                # Get current observation from the training_env (potentially normalized)
                # For VecNormalize, buf_obs holds the last observation.
                # If not VecNormalize, this might need adjustment or a direct call to get observation.
                obs_for_video_predict = self.training_env.buf_obs if hasattr(
                    self.training_env, 'buf_obs') else self.training_env.reset()

                ep_rewards_video = []
                # temp_terminated_or_truncated = False # This was for a single env, now assuming VecEnv
                dones_video = np.array([False] * self.training_env.num_envs)

                for _ in range(self.video_length):
                    if dones_video[0]:  # Check only the first env for video recording done
                        break
                    all_actions, _ = self.model.predict(
                        obs_for_video_predict, deterministic=True)
                    next_obs_dict, rewards, dones_step, infos = self.training_env.step(
                        all_actions)  # Step the wrapped (potentially normalized) env
                    obs_for_video_predict = next_obs_dict
                    # Log reward of the first env
                    ep_rewards_video.append(rewards[0])
                    dones_video = dones_step  # Update dones for all sub-environments

                video_file_path = actual_env_for_video.stop_video_recording(
                    save_dir=video_save_dir, filename=video_filename, fps=self.video_fps)

                if video_file_path and self.wandb_run:
                    try:
                        log_payload = {
                            "media/training_progress_video": wandb.Video(video_file_path, fps=self.video_fps, format="mp4"),
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


def setup_wandb(project_name: Optional[str], entity: Optional[str], config: Dict[str, Any],
                monitor_gym: bool = True, save_code: bool = True) -> Optional[WandbRun]:
    """ Sets up Weights & Biases. (Implementation remains the same) """
    if not project_name:
        return None
    run = None
    try:
        wandb.login()
    except Exception:
        try:
            key = getpass.getpass(
                prompt="Enter your W&B API key (or press Enter to skip W&B): ")
            if key:
                os.environ["WANDB_API_KEY"] = key
                wandb.login()
            else:
                return None
        except Exception as e:
            return None
    try:
        run = wandb.init(project=project_name, entity=entity, config=config,
                         sync_tensorboard=True, monitor_gym=monitor_gym, save_code=save_code)
    except Exception as e:
        run = None
    return run


def create_env_config(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """ Creates the configuration dictionary for FrankaShelfEnv. """
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
        # MOD: Added shelf optionality flags
        include_shelf=training_config.get('include_shelf', True),
        randomize_shelf_config=training_config.get(
            'randomize_shelf_config', True)
    )


def create_training_env(num_genesis_envs: int, seed: int, env_kwargs: Dict[str, Any], gamma: float) -> VecNormalize:
    """ Creates and wraps FrankaShelfEnv. """
    try:
        env = FrankaShelfEnv(**env_kwargs)
        env.seed(seed)
    except Exception as e:
        # Original error message
        print(f"CRITICAL ERROR: Failed to create FrankaShelfEnv: {e}")
        traceback.print_exc()  # Original traceback
        sys.exit(1)
    if not (hasattr(env, 'num_envs') and hasattr(env, 'observation_space') and hasattr(env, 'action_space') and
            callable(getattr(env, 'reset', None)) and callable(getattr(env, 'step_async', None)) and
            callable(getattr(env, 'step_wait', None))):
        # Original error message
        print("CRITICAL ERROR: Created environment does not conform to VecEnv interface.")
        sys.exit(1)
    if env.num_envs != num_genesis_envs:
        # Original did not have a print here, but it's fine to keep a warning if mismatch
        print(
            f"Warning: FrankaShelfEnv num_envs ({env.num_envs}) differs from requested num_genesis_envs ({num_genesis_envs}).")
    normalized_env = VecNormalize(
        env, norm_obs=True, norm_reward=True, gamma=gamma)
    return normalized_env


def create_ppo_model(config: Dict[str, Any], env: VecNormalize,
                     policy_kwargs: Optional[Dict[str, Any]], log_dir: str) -> PPO:
    """ Creates the PPO agent. """
    model = PPO(
        policy=config["policy_type"], env=env, verbose=1, tensorboard_log=log_dir,
        seed=config["seed"], gamma=config["gamma"], gae_lambda=config["gae_lambda"],
        n_steps=config["n_steps_ppo"], batch_size=config["batch_size_ppo"],
        n_epochs=config["n_epochs_ppo"], learning_rate=config["learning_rate"],
        clip_range=config["clip_range"], ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"], max_grad_norm=config["max_grad_norm"],
        policy_kwargs=policy_kwargs
    )
    return model


def create_callbacks(config: Dict[str, Any], model: PPO, eval_env: VecNormalize,
                     log_dir: str, model_save_path_base: str,
                     wandb_run_instance: Optional[WandbRun]) -> List[BaseCallback]:
    """ Creates training callbacks. """
    callbacks = []
    samples_per_ppo_rollout = model.n_steps * model.n_envs
    checkpoint_save_freq_steps = samples_per_ppo_rollout * \
        config["checkpoint_save_freq_rollouts"]
    eval_freq_steps = samples_per_ppo_rollout * config["eval_freq_rollouts"]

    video_checkpoint_cb = VideoCheckpointCallback(  # Uses the original VideoCheckpointCallback structure
        save_freq=checkpoint_save_freq_steps, save_path=log_dir,
        name_prefix=f"{os.path.basename(model_save_path_base)}_ckpt",
        video_log_freq_multiplier=config["video_log_freq_multiplier"],
        video_length=config["video_length"], video_fps=config["video_fps"],
        wandb_run=wandb_run_instance, verbose=1
    )
    callbacks.append(video_checkpoint_cb)

    eval_callback = EvalCallback(
        eval_env=eval_env, best_model_save_path=os.path.join(
            log_dir, "best_model"),
        log_path=log_dir, eval_freq=eval_freq_steps,
        n_eval_episodes=max(5 * eval_env.num_envs // 2,
                            eval_env.num_envs if eval_env.num_envs > 0 else 5),
        deterministic=True, render=False, verbose=1
    )
    callbacks.append(eval_callback)

    if wandb_run_instance:
        wandb_sb3_callback = WandbCallbackSB3(
            model_save_path=os.path.join(
                log_dir, f"wandb_models/{wandb_run_instance.id}"),
            model_save_freq=samples_per_ppo_rollout *
            # Original: was missing .get() default
            config.get("wandb_model_save_freq_rollouts"),
            # gradient_save_freq = 0, # Original did not have this line
            log="all", verbose=2  # Original log="all"
        )
        callbacks.append(wandb_sb3_callback)
    return callbacks


def train_and_save_model(model: PPO, config: Dict[str, Any], callbacks: List[BaseCallback],
                         model_save_path_base: str, env: VecNormalize, log_dir: str,
                         wandb_run_instance: Optional[WandbRun]) -> None:
    """ Runs training loop and handles final model saving. """
    try:
        model.learn(
            total_timesteps=config["total_timesteps"], callback=callbacks, progress_bar=True)
    except Exception as e:
        traceback.print_exc()  # Original had this
    finally:
        save_dir = os.path.dirname(model_save_path_base) if os.path.dirname(
            model_save_path_base) else log_dir
        final_model_name_stem = os.path.basename(model_save_path_base)
        os.makedirs(save_dir, exist_ok=True)

        full_final_model_path = os.path.join(
            save_dir, f"{final_model_name_stem}_final.zip")
        model.save(full_final_model_path)
        # print(f"Final model saved to {full_final_model_path}") # Original did not have this print

        vec_normalize_stats_path = os.path.join(
            save_dir, f"{final_model_name_stem}_vecnormalize.pkl")
        if isinstance(env, VecNormalize):
            env.save(vec_normalize_stats_path)
            # print(f"VecNormalize stats saved to {vec_normalize_stats_path}") # Original did not have this print

        if wandb_run_instance:
            try:
                trained_model_artifact = wandb.Artifact(
                    name=f'{final_model_name_stem}_final_model', type='model')
                trained_model_artifact.add_file(full_final_model_path)
                wandb_run_instance.log_artifact(trained_model_artifact)
                # print(f"Logged final model to W&B: {trained_model_artifact.name}") # Original did not have this

                if isinstance(env, VecNormalize) and os.path.exists(vec_normalize_stats_path):
                    vec_stats_artifact = wandb.Artifact(
                        name=f'{final_model_name_stem}_vecnormalize_stats', type='dataset')
                    vec_stats_artifact.add_file(vec_normalize_stats_path)
                    wandb_run_instance.log_artifact(vec_stats_artifact)
                    # print(f"Logged VecNormalize stats to W&B: {vec_stats_artifact.name}") # Original did not have this
            except Exception as e:
                # print(f"Error logging final artifacts to W&B: {e}") # Original did not have this
                pass


# MOD: env can be None
def cleanup_training(env: Optional[VecNormalize], wandb_run_instance: Optional[WandbRun]) -> None:
    """ Performs cleanup. """
    if env:
        env.close()  # Original: env.close()
        # print("Training environment closed.") # Original did not have this
    if wandb_run_instance:
        wandb_run_instance.finish()
        # print("W&B run finished.") # Original did not have this


def run_franka_training(training_config: Dict[str, Any]) -> None:
    """ Main orchestrator for the Franka agent training process. """
    gs_initialized_locally = False
    # MOD: Initialize to allow assignment in try
    train_env_instance: Optional[VecNormalize] = None
    # MOD: Initialize to allow assignment in try
    wandb_run: Optional[WandbRun] = None

    try:
        if not (hasattr(gs, '_is_initialized') and gs._is_initialized):
            # backend_to_use = gs.gpu if torch.cuda.is_available() else gs.cpu # Original logic
            # print(f"Initializing Genesis with backend: {'GPU' if backend_to_use == gs.gpu else 'CPU'}") # Original print
            gs.init(backend=gs.gpu if torch.cuda.is_available()
                    else gs.cpu)  # Original init
            gs_initialized_locally = True
            # print("Genesis initialized successfully.") # Original print

        set_random_seed(training_config["seed"])
        # print(f"Random seed set to: {training_config['seed']}") # Original print

        wandb_run = setup_wandb(  # MOD: Assign to wandb_run
            project_name=training_config.get("wandb_project_name"),
            entity=training_config.get("wandb_entity"),
            config=training_config
        )

        log_dir = training_config["log_dir"]
        model_save_path_base = training_config["model_save_path"]
        os.makedirs(log_dir, exist_ok=True)
        if os.path.dirname(model_save_path_base):
            os.makedirs(os.path.dirname(model_save_path_base), exist_ok=True)
        # print(f"Log directory: {log_dir}") # Original print
        # print(f"Base model save path: {model_save_path_base}") # Original print

        env_specific_config = create_env_config(training_config)
        if wandb_run:
            wandb_env_config_to_log = {
                f"env_{k}": v for k, v in env_specific_config.items() if k not in training_config}  # MOD: f"env_{k}"
            if wandb_env_config_to_log:
                wandb.config.update(wandb_env_config_to_log,
                                    allow_val_change=True)

        train_env_instance = create_training_env(  # MOD: Assign to train_env_instance
            num_genesis_envs=training_config["num_genesis_envs"],
            seed=training_config["seed"],
            env_kwargs=env_specific_config,
            gamma=training_config["gamma"]
        )
        # print("Training environment created and normalized.") # Original print

        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        # print(f"Using policy_kwargs: {policy_kwargs}") # Original print

        ppo_model = create_ppo_model(
            config=training_config,
            env=train_env_instance,  # MOD: Use train_env_instance
            policy_kwargs=policy_kwargs,
            log_dir=log_dir
        )
        # print(f"PPO model created with policy: {training_config['policy_type']}") # Original print

        training_callbacks = create_callbacks(
            # MOD: Use train_env_instance
            config=training_config, model=ppo_model, eval_env=train_env_instance,
            log_dir=log_dir, model_save_path_base=model_save_path_base,
            wandb_run_instance=wandb_run  # MOD: Use wandb_run
        )
        # print(f"Created {len(training_callbacks)} callbacks for training.") # Original print

        # print("Starting model training...") # Original print
        train_and_save_model(
            model=ppo_model, config=training_config, callbacks=training_callbacks,
            # MOD: Use train_env_instance
            model_save_path_base=model_save_path_base, env=train_env_instance,
            log_dir=log_dir, wandb_run_instance=wandb_run  # MOD: Use wandb_run
        )
        # print("Model training finished.") # Original print

    except Exception as e:  # Original had this
        traceback.print_exc()
    finally:
        # print("Performing cleanup...") # Original print
        # MOD: Pass initialized variables
        cleanup_training(train_env_instance, wandb_run)

        if gs_initialized_locally and hasattr(gs, 'shutdown') and callable(gs.shutdown):
            try:  # Original try-except for shutdown
                gs.shutdown()
                # print("Genesis shut down successfully.") # Original print
            except Exception as e:
                print(f"Error during Genesis shutdown: {e}")  # Original print
        # print("Training script finished.") # Original print


if __name__ == '__main__':
    config_params = {
        "policy_type": "MultiInputPolicy",
        "seed": 42,
        "gamma": 0.99, "gae_lambda": 0.95, "n_steps_ppo": 64,
        "batch_size_ppo": 2048, "n_epochs_ppo": 10, "learning_rate": 5e-4,
        "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,

        "total_timesteps": 10_000_000,
        "num_genesis_envs": 1024,
        "workspace_bounds_xyz_override": ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.5)),

        # MOD: Added shelf optionality flags to default config
        "include_shelf": True,
        "randomize_shelf_config": True,

        "log_dir": "./training_logs/ppo_franka_shelf_params_obs/",
        "model_save_path": "./training_logs/ppo_franka_shelf_params_obs/model_shelf_params",
        # MOD: Can be changed for different runs
        "wandb_project_name": "FrankaPPO-ShelfParamsObs",
        "wandb_entity": None,
        "wandb_model_save_freq_rollouts": 20,
        "checkpoint_save_freq_rollouts": 10,
        "eval_freq_rollouts": 20,
        "video_log_freq_multiplier": 2,
        "video_length": 500, "video_fps": 30,

        "k_dist_reward": 20.0, "k_time_penalty": 0.02, "k_action_penalty": 0.005,
        "k_joint_limit_penalty": 15.0, "k_collision_penalty": 50.0, "k_accel_penalty": 0.005,
        "success_reward_val": 350.0, "success_threshold_val": 0.05,
        "max_steps_per_episode": 750,
        'video_camera_pos_override': (1.8, -1.8, 2.0),
        'video_camera_lookat_override': (0.3, 0.0, 0.5),
        'video_camera_fov_override': 45,
        'video_res_override': (960, 640)
    }

    # Example of how to run different configurations:
    # To run without shelf:
    config_params["include_shelf"] = False
    # config_params["randomize_shelf_config"] = False # Does not matter if shelf is not included
    # config_params["wandb_project_name"] = "FrankaPPO-NoShelf"
    # config_params["log_dir"] = "./training_logs/ppo_franka_no_shelf/"
    # config_params["model_save_path"] = "./training_logs/ppo_franka_no_shelf/model_no_shelf"

    # To run with shelf but not randomized (fixed position):
    # config_params["include_shelf"] = True
    # config_params["randomize_shelf_config"] = False
    # config_params["wandb_project_name"] = "FrankaPPO-FixedShelf"
    # config_params["log_dir"] = "./training_logs/ppo_franka_fixed_shelf/"
    # config_params["model_save_path"] = "./training_logs/ppo_franka_fixed_shelf/model_fixed_shelf"

    run_franka_training(training_config=config_params)
