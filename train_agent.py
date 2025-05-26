# <filename>train_agent.py</filename>
import os
import sys
import torch
import genesis as gs  # type: ignore
import wandb
import getpass
# Added Optional, List, Any, Tuple, Dict here
from typing import Optional, List, Any, Tuple, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed, safe_mean
from wandb.integration.sb3 import WandbCallback
import numpy as np  # For video recording loop

from franka_rl_env import FrankaShelfEnv  # Assuming FrankaShelfEnv is VecEnv
from collision_cnn import CustomCNNFeatureExtractor


class VideoCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model checkpoint and recording/logging a video of the agent.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model",
                 # Record video every 'video_log_freq_multiplier' checkpoints
                 video_log_freq_multiplier: int = 1,
                 video_length: int = 200, video_fps: int = 30, verbose: int = 0,
                 wandb_run=None):  # Pass wandb run object
        super().__init__(save_freq=save_freq, save_path=save_path,
                         name_prefix=name_prefix, verbose=verbose)
        self.video_length = video_length
        self.video_fps = video_fps
        self.wandb_run = wandb_run
        self.video_log_freq_multiplier = video_log_freq_multiplier
        self.checkpoint_count = 0  # To control video logging frequency

    def _on_step(self) -> bool:
        # Call CheckpointCallback's _on_step first
        continue_training = super()._on_step()
        if not continue_training:
            return False

        if self.num_timesteps % self.save_freq == 0:
            self.checkpoint_count += 1
            if self.checkpoint_count % self.video_log_freq_multiplier == 0:
                if self.verbose > 0:
                    print(
                        f"Saving model checkpoint and recording video at timestep {self.num_timesteps}")

                actual_env = self.training_env.venv
                if isinstance(actual_env, DummyVecEnv):
                    if hasattr(actual_env.envs[0], 'start_video_recording'):
                        env_to_record = actual_env.envs[0]
                    else:
                        if self.verbose > 0:
                            print(
                                "Warning: Could not get underlying FrankaShelfEnv for video recording from DummyVecEnv.")
                        return True
                elif hasattr(actual_env, 'start_video_recording'):
                    env_to_record = actual_env
                else:
                    if self.verbose > 0:
                        print(
                            "Warning: Could not get underlying FrankaShelfEnv for video recording.")
                    return True

                video_filename = f"{self.name_prefix}_{self.num_timesteps}_video.mp4"
                video_save_dir = os.path.join(self.save_path, "videos")

                if not env_to_record.start_video_recording(env_idx_to_focus=0):
                    if self.verbose > 0:
                        print("Failed to start video recording in environment.")
                    return True

                if self.verbose > 0:
                    print(
                        f"Running dedicated episode for video recording (length {self.video_length})...")

                obs_dict = self.training_env.reset()
                current_obs_env0 = {k: v[0:1] for k, v in obs_dict.items()}

                ep_rewards_video = []
                temp_terminated = False

                for _frame_num in range(self.video_length):
                    if temp_terminated:
                        break

                    all_actions, _ = self.model.predict(
                        obs_dict, deterministic=True)
                    obs_dict, rewards, dones, infos = self.training_env.step(
                        all_actions)

                    ep_rewards_video.append(rewards[0])
                    if dones[0]:
                        temp_terminated = True

                video_file_path = env_to_record.stop_video_recording(
                    save_dir=video_save_dir, filename=video_filename, fps=self.video_fps)

                if video_file_path and self.wandb_run:
                    try:
                        self.wandb_run.log({
                            "media/training_progress_video": wandb.Video(video_file_path, fps=self.video_fps, format="mp4"),
                            f"diagnostics/video_ep_mean_reward_env0_{self.num_timesteps}": safe_mean(ep_rewards_video) if ep_rewards_video else 0
                        }, step=self.num_timesteps)
                        if self.verbose > 0:
                            print(f"Logged video {video_file_path} to W&B.")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Error logging video to W&B: {e}")
        return True


def train_franka_agent(total_timesteps=1_000_000,
                       num_genesis_envs=4,
                       seed=42,
                       log_dir="./ppo_franka_logs/",
                       model_save_path="./ppo_franka_model",
                       wandb_project_name="FrankaShelfPPO",
                       wandb_entity=None,
                       use_cnn=True,
                       config_overrides: Optional[dict] = None):
    """
    Trains a PPO agent on FrankaShelfEnv.
    """
    if not hasattr(gs, '_is_initialized') or not gs._is_initialized:
        backend_to_use = gs.gpu if torch.cuda.is_available() else gs.cpu
        try:
            gs.init(backend=backend_to_use)
            print(
                f"Genesis initialized with backend: {'gpu' if backend_to_use == gs.gpu else 'cpu'}.")
        except Exception as e:
            print(
                f"CRITICAL ERROR: Failed to initialize Genesis: {e}\nExiting.")
            sys.exit(1)
    else:
        print("Genesis already initialized.")

    run = None
    if wandb_project_name:
        try:
            wandb.login(anonymous="allow")
        except Exception:
            print(
                "Could not log in to W&B automatically. Ensure WANDB_API_KEY is set or run 'wandb login'.")
            try:
                key = getpass.getpass(
                    prompt="Enter your W&B API key (or press Enter to skip W&B): ")
                if key:
                    os.environ["WANDB_API_KEY"] = key
                    wandb.login()
                else:
                    print("Skipping W&B logging as no API key was provided.")
                    wandb_project_name = None
            except Exception as e:
                print(
                    f"Failed to log in to W&B with provided key: {e}. Skipping W&B.")
                wandb_project_name = None
    else:
        print("W&B project name not specified, skipping W&B logging.")

    config = {
        "policy_type": "MultiInputPolicy", "total_timesteps": total_timesteps,
        "num_genesis_envs": num_genesis_envs, "seed": seed, "gamma": 0.99, "gae_lambda": 0.95,
        "n_steps_ppo": 2048,
        "batch_size_ppo": 64, "n_epochs_ppo": 10, "learning_rate": 3e-4,
        "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "use_cnn": use_cnn, "log_dir": log_dir, "model_save_path": model_save_path,
        "parallelism_mode": "genesis_direct_vecenv",
        "video_log_freq_multiplier": 5,
        "video_length": 250,
        "video_fps": 30,
        "checkpoint_save_freq_rollouts": 10,
        "eval_freq_rollouts": 5
    }
    if use_cnn:
        config["cnn_features_dim"] = 256

    if config_overrides:
        config.update(config_overrides)

    if wandb_project_name:
        run = wandb.init(project=wandb_project_name, entity=wandb_entity, config=config,
                         sync_tensorboard=True, monitor_gym=True, save_code=True)
        print(
            f"W&B run initialized: {run.url if run else 'None (failed init after login attempt?)'}")
    else:
        run = None
        print("W&B logging is disabled for this run.")

    os.makedirs(log_dir, exist_ok=True)
    if os.path.dirname(model_save_path):
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    set_random_seed(seed)

    env_kwargs_for_franka = dict(
        render_mode=None,
        num_envs=num_genesis_envs,
        k_dist_reward=config.get('k_dist_reward', 1.5),
        k_time_penalty=config.get('k_time_penalty', 0.02),
        k_action_penalty=config.get('k_action_penalty', 0.005),
        k_joint_limit_penalty=config.get('k_joint_limit_penalty', 15.0),
        k_collision_penalty=config.get('k_collision_penalty', 250.0),
        k_accel_penalty=config.get('k_accel_penalty', 0.005),
        success_reward_val=config.get('success_reward_val', 300.0),
        success_threshold_val=config.get('success_threshold_val', 0.05),
        max_steps_per_episode=config.get('max_steps_per_episode', 1000),
        video_camera_pos=tuple(config.get(
            'video_camera_pos_override', (1.8, -1.8, 2.0))),
        video_camera_lookat=tuple(config.get(
            'video_camera_lookat_override', (0.3, 0.0, 0.5))),
        video_camera_fov=config.get('video_camera_fov_override', 45),
        video_res=tuple(config.get('video_res_override', (320, 240)))
    )

    if run:
        wandb_env_config_to_log = {
            k: v for k, v in env_kwargs_for_franka.items() if k not in config}
        if wandb_env_config_to_log:
            wandb.config.update(wandb_env_config_to_log, allow_val_change=True)

    try:
        env = FrankaShelfEnv(**env_kwargs_for_franka)
        env.seed(seed)
    except Exception as e:
        print(f"Error initializing FrankaShelfEnv: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not (hasattr(env, 'num_envs') and hasattr(env, 'observation_space') and hasattr(env, 'action_space') and callable(getattr(env, 'reset', None)) and callable(getattr(env, 'step_async', None)) and callable(getattr(env, 'step_wait', None))):
        print("ERROR: FrankaShelfEnv does not seem to have the full VecEnv interface.")
        sys.exit(1)
    if env.num_envs != num_genesis_envs:
        print(
            f"Warning: FrankaShelfEnv.num_envs ({env.num_envs}) != requested ({num_genesis_envs}).")

    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       gamma=config["gamma"])
    print(
        f"FrankaShelfEnv (VecEnv with {env.num_envs} internal envs) wrapped with VecNormalize.")

    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=config.get("cnn_features_dim", 256))
    ) if config["use_cnn"] else None

    ppo_n_steps_val = config["n_steps_ppo"]

    model = PPO(
        config["policy_type"], env, verbose=1,
        tensorboard_log=log_dir,
        seed=seed, gamma=config["gamma"], gae_lambda=config["gae_lambda"],
        n_steps=ppo_n_steps_val, batch_size=config["batch_size_ppo"], n_epochs=config["n_epochs_ppo"],
        learning_rate=config["learning_rate"], clip_range=config["clip_range"],
        ent_coef=config["ent_coef"], vf_coef=config["vf_coef"], max_grad_norm=config["max_grad_norm"],
        policy_kwargs=policy_kwargs
    )
    print(
        f"PPO model created. Rollout buffer size: {model.n_steps} per env * {model.n_envs} envs = {model.n_steps * model.n_envs} total samples per update.")

    samples_per_ppo_rollout = model.n_steps * model.n_envs

    checkpoint_save_freq_steps = samples_per_ppo_rollout * \
        config["checkpoint_save_freq_rollouts"]
    eval_freq_steps = samples_per_ppo_rollout * config["eval_freq_rollouts"]

    print(
        f"Checkpoint save frequency: every {checkpoint_save_freq_steps} total env steps.")
    print(f"Evaluation frequency: every {eval_freq_steps} total env steps.")

    callbacks = []
    video_checkpoint_cb = VideoCheckpointCallback(
        save_freq=checkpoint_save_freq_steps,
        save_path=log_dir,
        name_prefix=f"{os.path.basename(model_save_path)}_ckpt",
        video_log_freq_multiplier=config["video_log_freq_multiplier"],
        video_length=config["video_length"],
        video_fps=config["video_fps"],
        wandb_run=run,
        verbose=1
    )
    callbacks.append(video_checkpoint_cb)

    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq_steps,
        n_eval_episodes=max(5 * env.num_envs // 2,
                            env.num_envs if env.num_envs > 0 else 5),
        deterministic=True, render=False, verbose=1
    )
    callbacks.append(eval_callback)

    if run:
        wandb_callback = WandbCallback(
            model_save_path=os.path.join(log_dir, f"wandb_models/{run.id}"),
            model_save_freq=samples_per_ppo_rollout *
            config.get("wandb_model_save_freq_rollouts", 20),
            log="all", verbose=2,
        )
        callbacks.append(wandb_callback)

    print(
        f"Starting training for {config['total_timesteps']} total environment steps...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        final_model_name = os.path.basename(model_save_path.rstrip('/\\'))
        save_dir = os.path.dirname(model_save_path) or log_dir
        full_final_model_path = os.path.join(
            save_dir, f"{final_model_name}_final.zip")
        model.save(full_final_model_path)
        print(f"Final model saved to {full_final_model_path}")

        vec_normalize_stats_path = os.path.join(
            save_dir, f"{final_model_name}_vecnormalize.pkl")
        if isinstance(env, VecNormalize):
            env.save(vec_normalize_stats_path)
            print(f"VecNormalize stats saved to {vec_normalize_stats_path}")

        env.close()
        print("Training environment closed.")

        if run:
            print("Logging final model and VecNormalize stats to W&B...")
            try:
                trained_model_artifact = wandb.Artifact(
                    f'{final_model_name}_final_model', type='model')
                trained_model_artifact.add_file(full_final_model_path)
                run.log_artifact(trained_model_artifact)
                if isinstance(env, VecNormalize) and os.path.exists(vec_normalize_stats_path):
                    vec_stats_artifact = wandb.Artifact(
                        f'{final_model_name}_vecnormalize_stats', type='dataset')
                    vec_stats_artifact.add_file(vec_normalize_stats_path)
                    run.log_artifact(vec_stats_artifact)
                print("Artifacts logged to W&B.")
            except Exception as e:
                print(f"Error logging artifacts to W&B: {e}")
            run.finish()
            print("W&B run finished.")

        if hasattr(gs, 'shutdown') and callable(gs.shutdown):
            print("Shutting down Genesis...")
            gs.shutdown()
        print("Training script finished.")


if __name__ == '__main__':
    config_params = {
        "total_timesteps": 1_000_000_000,
        "num_genesis_envs": 1024,
        "n_steps_ppo": 24,
        "seed": 42,
        "log_dir": "./training_logs/ppo_franka_shelf_video_v2/",
        "model_save_path": "./training_logs/ppo_franka_shelf_video_v2/ppo_franka_shelf_video_v2",
        "wandb_project_name": "FrankaShelfPPO-Video-v2",
        "wandb_entity": None,
        "use_cnn": True,
        "cnn_features_dim": 256,
        "checkpoint_save_freq_rollouts": 3,
        "eval_freq_rollouts": 3,
        "video_log_freq_multiplier": 10,
        "video_length": 300,
        "video_fps": 30,
        "k_collision_penalty": 200.0,
        "success_reward_val": 350.0,
        "max_steps_per_episode": 750
    }

    print("--- Franka Shelf PPO Training (Video Checkpoints, W&B) ---")
    for key, val in config_params.items():
        print(f"{key}: {val}")
    print("---------------------------------")

    train_franka_agent(
        total_timesteps=config_params["total_timesteps"],
        num_genesis_envs=config_params["num_genesis_envs"],
        seed=config_params["seed"],
        log_dir=config_params["log_dir"],
        model_save_path=config_params["model_save_path"],
        wandb_project_name=config_params["wandb_project_name"],
        wandb_entity=config_params["wandb_entity"],
        use_cnn=config_params["use_cnn"],
        config_overrides=config_params
    )
