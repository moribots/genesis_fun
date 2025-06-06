"""
Main script for training a PPO agent on the FrankaShelfEnv environment using RSL-RL.

This script handles:
- Initialization of the Genesis simulation environment.
- Configuration of the training environment and the RSL-RL PPO algorithm.
- Creation of the FrankaShelfEnv vectorized environment.
- Instantiation and execution of a custom OnPolicyRunner, which manages the
  entire training loop, including logging to Weights & Biases, checkpointing,
  and periodic video recording.
"""
import os
import sys
import traceback
from dataclasses import dataclass, asdict, fields
import time
from collections import deque
import numpy as np
import statistics

import torch
import genesis as gs  # type: ignore
from torch.utils.tensorboard import SummaryWriter
import imageio
import wandb

from franka_rl_env import FrankaShelfEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.utils.wandb_utils import WandbSummaryWriter


class CustomOnPolicyRunner(OnPolicyRunner):
    """
    A custom runner that inherits from RSL-RL's OnPolicyRunner to add
    periodic video recording and logging to Weights & Biases.
    """

    def __init__(self, env, train_cfg: 'TrainConfig', log_dir, device='cpu'):
        """
        Initializes the custom runner.

        This method bridges the gap between the script's dataclass-based configuration
        and the RSL-RL runner's expectation of a dictionary-based configuration.

        Args:
            env: The vectorized environment to train on.
            train_cfg: The master configuration object (a TrainConfig dataclass).
            log_dir: The directory to save logs and checkpoints.
            device: The device to run the training on ('cpu' or 'cuda').
        """
        # RSL-RL's OnPolicyRunner expects a dictionary where runner parameters are at the top level,
        # and policy/algorithm parameters are in nested dictionaries. We manually construct this
        # dictionary from our dataclass structure to ensure correctness.

        # Start with the nested dictionaries for policy and algorithm
        runner_internal_cfg = {
            "policy": asdict(train_cfg.policy),
            "algorithm": asdict(train_cfg.algorithm)
        }
        # Add all fields from the RunnerConfig dataclass to the top level of the dictionary.
        # This will include 'wandb_entity' and 'wandb_project', which are needed by WandbSummaryWriter.
        for field in fields(train_cfg.runner):
            runner_internal_cfg[field.name] = getattr(
                train_cfg.runner, field.name)

        # Call the parent constructor with the explicitly constructed dictionary
        super().__init__(env, runner_internal_cfg, log_dir, device)

        # Store the original dataclass config for convenient access in this custom class
        self.full_cfg = train_cfg

        # Now, access video settings from the stored dataclass config
        self.video_log_interval = self.full_cfg.runner.video_log_interval
        self.video_length = self.full_cfg.runner.video_length
        self.last_video_log_time = 0

    def log_video(self, it):
        """
        Records and logs a video of the policy interacting with the environment.
        """
        if not self.writer or not hasattr(self.env, 'start_video_recording'):
            return

        print("--- Recording video ---")
        video_name = f'iteration_{it}.mp4'
        video_path = os.path.join(self.log_dir, video_name)

        # Start recording
        self.env.start_video_recording()

        # Run a short episode for the video
        obs, infos = self.env.get_observations()
        critic_obs = infos.get("observations", {}).get("critic", obs)
        for _ in range(self.video_length):
            with torch.no_grad():
                actions = self.alg.act(obs, critic_obs)
            obs, privileged_obs, rews, dones, infos = self.env.step(actions)
            critic_obs = infos.get("observations", {}).get("critic", obs)

        # Stop recording and save the video
        self.env.stop_video_recording(video_path)
        print(f"--- Video saved to {video_path} ---")

        # Log video
        try:
            # Check logger type to use the correct logging method
            if isinstance(self.writer, WandbSummaryWriter):
                # W&B can log video directly from a path
                wandb.log({"policy_rollout": wandb.Video(
                    video_path, fps=30, format="mp4")}, step=it)
                print("--- Logged video to W&B ---")
            else:  # Default to Tensorboard SummaryWriter
                # Tensorboard needs the video as a tensor
                video_reader = imageio.get_reader(video_path)
                fps = video_reader.get_meta_data()['fps']
                # From (T, H, W, C) to (N, T, C, H, W) where N=1
                video_frames = np.array([frame for frame in video_reader])
                video_tensor = torch.from_numpy(
                    video_frames).permute(0, 3, 1, 2).unsqueeze(0)
                self.writer.add_video(
                    tag="policy_rollout", vid_tensor=video_tensor, global_step=it, fps=fps)
                print("--- Logged video to Tensorboard ---")
        except Exception as e:
            print(f"Error logging video: {e}")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        return self.alg.actor_critic.act_inference

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        Overrides the main learn method to add video logging and correct timing.
        """
        # Initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "wandb":
                # The W&B run is already initialized. WandbSummaryWriter's init will
                # just grab the existing run object.
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir)
            else:
                raise ValueError(
                    f"Logger type {self.logger_type} not supported")

        if hasattr(self, 'fixed_games_to_play') and self.fixed_games_to_play is not None:
            self.env.start_recording()

        obs, infos = self.env.get_observations()
        critic_obs = infos.get("observations", {}).get("critic", obs)

        self.alg.actor_critic.train()

        # Bookkeeping for logging, mirrored from the base runner
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        # Add buffer for the new target_reached metric
        target_reached_buffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device)

        # Bookkeeping for custom reward component logging
        reward_component_names = [
            "distance", "time_penalty", "action_penalty", "joint_limit_penalty",
            "collision_penalty", "success_reward", "accel_penalty", "alive_bonus",
            "joint_velocity_penalty"
        ]
        reward_component_buffers = {name: deque(
            maxlen=100) for name in reward_component_names}
        current_reward_component_sums = {
            name: torch.zeros(self.env.num_envs,
                              dtype=torch.float, device=self.device)
            for name in reward_component_names
        }

        start_iter = self.current_learning_iteration
        # Define tot_iter for the logging function
        tot_iter = start_iter + num_learning_iterations
        tot_time = 0

        for it in range(start_iter, tot_iter):
            # --- Data Collection Phase ---
            collection_start = time.time()

            # List to store episode lengths from the current iteration
            iteration_episode_lengths = []
            iteration_target_reached = []

            with torch.inference_mode():
                # Rollout over the environment
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rews, dones, infos = self.env.step(
                        actions)
                    critic_obs = infos.get(
                        "observations", {}).get("critic", obs)

                    self.alg.process_env_step(rews, dones, infos)

                    # Update standard and custom reward/length buffers
                    cur_reward_sum += rews
                    cur_episode_length += 1
                    for name in reward_component_names:
                        if f"reward_components/{name}" in infos:
                            current_reward_component_sums[
                                name] += infos[f"reward_components/{name}"]

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        # Get lengths of episodes that just finished in this step
                        finished_lengths = cur_episode_length[new_ids][:,
                                                                       0].cpu().numpy().tolist()
                        iteration_episode_lengths.extend(finished_lengths)
                        if "is_target_reached" in infos:
                            # Convert booleans to integers (0s and 1s) for statistics
                            iteration_target_reached.extend(
                                infos["is_target_reached"][new_ids][:, 0].int().cpu().numpy().tolist())

                        # Update main buffers for the base logger and smoothed stats
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(finished_lengths)
                        if iteration_target_reached:
                            target_reached_buffer.extend(
                                iteration_target_reached)

                        # Reset buffers for these envs
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        for name, sums in current_reward_component_sums.items():
                            reward_component_buffers[name].extend(
                                sums[new_ids][:, 0].cpu().numpy().tolist())
                            sums[new_ids] = 0

                # Check for video logging interval during collection
                if self.video_log_interval > 0 and (it % self.video_log_interval == 0):
                    self.log_video(it)

                # Compute returns before learning
                self.alg.compute_returns(critic_obs)

            collection_time = time.time() - collection_start

            # --- Learning Phase ---
            learn_start = time.time()
            # Unpack all 5 values returned by update to prevent future KeyErrors
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
            learn_time = time.time() - learn_start

            # Set current iteration for logging and saving
            self.current_learning_iteration = it

            # Retrieve episode infos before logging
            ep_infos = self.env.get_episode_infos()

            # Base logging from the parent class
            if self.log_dir is not None:
                self.log(locals())

            # --- Custom detailed logging ---
            if self.log_dir is not None and self.writer is not None:
                # Explained variance
                explained_var = 1 - torch.var(self.alg.storage.values -
                                              self.alg.storage.returns) / torch.var(self.alg.storage.returns)
                self.writer.add_scalar(
                    "Train/explained_variance", explained_var.item(), it)

                # KL divergence (if available from storage)
                if hasattr(self.alg.storage, 'kl'):
                    mean_kl = self.alg.storage.kl.mean()
                    self.writer.add_scalar(
                        "Train/kl_divergence", mean_kl.item(), it)

                # Log individual reward components
                for name, buffer in reward_component_buffers.items():
                    if len(buffer) > 0:
                        self.writer.add_scalar(
                            f"Rewards/{name}", statistics.mean(buffer), it)

                # Log episode length stats from the current iteration only
                if iteration_episode_lengths:
                    self.writer.add_scalar(
                        "Rollout/length_min", min(iteration_episode_lengths), it)
                    self.writer.add_scalar(
                        "Rollout/length_mean", statistics.mean(iteration_episode_lengths), it)
                    self.writer.add_scalar(
                        "Rollout/length_max", max(iteration_episode_lengths), it)

                # Log target reached rate for the current rollout
                if iteration_target_reached:
                    self.writer.add_scalar(
                        "Rollout/target_reached_rate", statistics.mean(iteration_target_reached), it)

                # Log curriculum-related info from the env's `extras` dict (now in `infos`)
                if "curriculum/success_rate" in infos:
                    self.writer.add_scalar(
                        "Curriculum/Success Rate", infos["curriculum/success_rate"].item(), it)
                if "curriculum/target_reached_rate" in infos:
                    self.writer.add_scalar(
                        "Curriculum/Target Reached Rate", infos["curriculum/target_reached_rate"].item(), it)
                if "curriculum/time_penalty_scale" in infos:
                    self.writer.add_scalar(
                        "Curriculum/Time Penalty Scale", infos["curriculum/time_penalty_scale"].item(), it)
                if "curriculum/alive_bonus_scale" in infos:
                    self.writer.add_scalar(
                        "Curriculum/Alive Bonus Scale", infos["curriculum/alive_bonus_scale"].item(), it)
                if "curriculum/joint_vel_penalty_scale" in infos:
                    self.writer.add_scalar(
                        "Curriculum/Joint Vel Penalty Scale", infos["curriculum/joint_vel_penalty_scale"].item(), it)

            # Checkpoint saving
            if self.save_interval > 0 and (it % self.save_interval == 0):
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))

            tot_time += collection_time + learn_time

        # Save final model
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(
            self.current_learning_iteration)))

# --- RSL-RL Configuration ---


@dataclass
class EnvConfig:
    """Configuration for the FrankaShelfEnv environment."""
    num_envs: int = 3072
    num_obs: int = 24  # robot_state (21) + relative_target (3)
    num_privileged_obs: int = 0
    num_actions: int = 7
    max_episode_length: int = 300
    seed: int = 42

    # Environment-specific parameters, passed to FrankaShelfEnv constructor
    k_dist_reward: float = 2.0
    k_time_penalty: float = 0.0
    k_action_penalty: float = 0.0005
    k_joint_limit_penalty: float = 5.0
    k_collision_penalty: float = 20.0
    k_accel_penalty: float = 0.0001
    k_alive_bonus: float = 0.1
    k_joint_velocity_penalty: float = 2.0
    success_reward_val: float = 300.0
    success_threshold_val: float = 0.05
    settle_time_in_steps: int = 20

    # Curriculum Learning Parameters
    success_rate_threshold: float = 0.7
    target_reached_rate_threshold: float = 0.9
    curriculum_transition_width: float = 0.1
    min_episode_length_for_success_metric: int = 10

    include_shelf: bool = False
    randomize_shelf_config: bool = True
    workspace_bounds_xyz: tuple = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 1.5))
    video_camera_pos: tuple = (1.8, -1.8, 2.0)
    video_camera_lookat: tuple = (0.3, 0.0, 0.5)
    video_camera_fov: float = 45
    video_res: tuple = (960, 640)


@dataclass
class PolicyConfig:
    """Configuration for the Actor-Critic policy network."""
    class_name: str = 'ActorCritic'
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None
    activation: str = 'elu'
    init_noise_std: float = 1.0

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [512, 256, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 512, 256]


@dataclass
class AlgorithmConfig:
    """Configuration for the PPO algorithm."""
    class_name: str = 'PPO'
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.0015
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class RunnerConfig:
    """Configuration for the RSL-RL OnPolicyRunner."""
    # --- General ---
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    max_iterations: int = 2001

    # --- Runner-specific parameters for RSL-RL ---
    empirical_normalization: bool = True
    num_steps_per_env: int = 24
    logger: str = "wandb"

    # --- Logging and Saving ---
    log_dir: str = "./training_logs/rsl_ppo_franka/"
    run_name: str = ""  # If empty, a unique name will be generated
    experiment_name: str = "trajectory-tracking-franka"
    save_interval: int = 100  # in iterations
    checkpoint_path: str = ""  # path to checkpoint

    # --- Video Logging ---
    video_log_interval: int = 500  # log video every 1000 iterations
    video_length: int = 300  # steps to record in video

    # --- W&B Integration ---
    wandb: bool = True
    wandb_project: str = "franka-trajectory-tracking"
    wandb_entity: str = None  # Set to None to be fetched automatically
    wandb_group: str = ""
    wandb_tags: list = None

    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["rsl_rl", "franka",
                               "reach", "trajectory-tracking"]


@dataclass
class TrainConfig:
    """Top-level configuration container."""
    env: EnvConfig = EnvConfig()
    policy: PolicyConfig = PolicyConfig()
    runner: RunnerConfig = RunnerConfig()
    algorithm: AlgorithmConfig = AlgorithmConfig()


def run_franka_training(cfg: TrainConfig) -> None:
    """
    Main orchestrator for the Franka agent training process.
    Initializes Genesis, sets up the environment and RSL-RL runner,
    and starts the training.

    :param cfg: A TrainConfig object containing all configuration parameters.
    """
    gs_initialized_locally = False
    env = None
    wandb_run = None

    try:
        # --- Handle W&B Login, config, and Initialization ---
        if cfg.runner.wandb:
            try:
                # Ensure user is logged in to W&B
                if wandb.api.api_key is None:
                    wandb.login()

                # Fetch the default entity from the W&B API if not provided
                if cfg.runner.wandb_entity is None:
                    api = wandb.Api()
                    default_entity = api.default_entity
                    if default_entity:
                        print(
                            f"--- W&B entity not provided, using default: {default_entity} ---")
                        cfg.runner.wandb_entity = default_entity
                    else:
                        raise ValueError(
                            "Could not determine W&B default entity. Please set it manually in the config.")

                # Initialize the W&B run here to ensure it's done correctly before the runner starts
                wandb_run = wandb.init(
                    project=cfg.runner.wandb_project,
                    entity=cfg.runner.wandb_entity,
                    group=cfg.runner.wandb_group or None,
                    name=cfg.runner.run_name or None,
                    tags=cfg.runner.wandb_tags,
                    config=asdict(cfg)  # Log the entire configuration
                )

            except Exception as e:
                print(f"--- W&B setup failed: {e}. Disabling W&B logging. ---")
                cfg.runner.wandb = False
                if wandb_run:
                    wandb_run.finish()

        # --- Initialize Genesis ---
        if not (hasattr(gs, '_is_initialized') and gs._is_initialized):
            print(
                f"Initializing Genesis with backend: {cfg.runner.device.upper()}")
            backend_to_use = gs.gpu if cfg.runner.device == 'cuda' else gs.cpu
            gs.init(backend=backend_to_use)
            gs_initialized_locally = True
            print("Genesis initialized successfully.")

        # --- Create Environment ---
        env_kwargs = {
            "num_envs": cfg.env.num_envs,
            "max_steps_per_episode": cfg.env.max_episode_length,
            "k_dist_reward": cfg.env.k_dist_reward,
            "k_time_penalty": cfg.env.k_time_penalty,
            "k_action_penalty": cfg.env.k_action_penalty,
            "k_joint_limit_penalty": cfg.env.k_joint_limit_penalty,
            "k_collision_penalty": cfg.env.k_collision_penalty,
            "k_accel_penalty": cfg.env.k_accel_penalty,
            "k_alive_bonus": cfg.env.k_alive_bonus,
            "k_joint_velocity_penalty": cfg.env.k_joint_velocity_penalty,
            "success_reward_val": cfg.env.success_reward_val,
            "success_threshold_val": cfg.env.success_threshold_val,
            "settle_time_in_steps": cfg.env.settle_time_in_steps,
            "success_rate_threshold": cfg.env.success_rate_threshold,
            "target_reached_rate_threshold": cfg.env.target_reached_rate_threshold,
            "curriculum_transition_width": cfg.env.curriculum_transition_width,
            "min_episode_length_for_success_metric": cfg.env.min_episode_length_for_success_metric,
            "include_shelf": cfg.env.include_shelf,
            "randomize_shelf_config": cfg.env.randomize_shelf_config,
            "workspace_bounds_xyz": cfg.env.workspace_bounds_xyz,
            "video_camera_pos": cfg.env.video_camera_pos,
            "video_camera_lookat": cfg.env.video_camera_lookat,
            "video_camera_fov": cfg.env.video_camera_fov,
            "video_res": cfg.env.video_res,
            "device": cfg.runner.device,
            "seed": cfg.env.seed
        }
        env = FrankaShelfEnv(**env_kwargs)
        print("FrankaShelfEnv created successfully.")

        # --- Create Log Directory ---
        os.makedirs(cfg.runner.log_dir, exist_ok=True)

        # --- Instantiate and Run CustomOnPolicyRunner ---
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
            print("Environment closed.")

        # Ensure the W&B run is properly finished
        if wandb_run:
            wandb_run.finish()
            print("W&B run finished.")

        if gs_initialized_locally and hasattr(gs, 'shutdown') and callable(gs.shutdown):
            try:
                gs.shutdown()
                print("Genesis shut down successfully.")
            except Exception as e:
                print(f"Error during Genesis shutdown: {e}")
        print("Training script finished.")


if __name__ == '__main__':
    training_config = TrainConfig()
    run_franka_training(training_config)
