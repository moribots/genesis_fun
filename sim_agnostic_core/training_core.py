"""
This script contains the core, simulator-agnostic components for training a PPO
agent using RSL-RL. It defines:
- Dataclasses for configuring the environment, policy, algorithm, and runner.
- A custom `OnPolicyRunner` that adds periodic video recording and enhanced
  logging capabilities.
"""
import os
import time
import statistics
from collections import deque
from dataclasses import dataclass, asdict, fields, field

import torch
import numpy as np
import imageio
import wandb
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.wandb_utils import WandbSummaryWriter

from sim_agnostic_core.curriculum_core import CurriculumConfig


class CustomOnPolicyRunner(OnPolicyRunner):
    """
    A custom runner that extends RSL-RL's OnPolicyRunner.

    This class adds two key features:
    1.  **Periodic Video Recording**: Automatically records and logs videos of
        the agent's performance to W&B or TensorBoard.
    2.  **Enhanced Logging**: Logs detailed, decomposed reward components,
        curriculum progress, and other useful metrics for in-depth analysis
        of the training process.
    """

    def __init__(self, env, train_cfg: 'TrainConfig', log_dir, device='cpu'):
        """
        Initializes the custom runner.

        This constructor adapts the dataclass-based configuration (`TrainConfig`)
        to the dictionary-based configuration expected by the base `OnPolicyRunner`.

        Args:
            env: The vectorized environment to train on.
            train_cfg: The master configuration object (a TrainConfig dataclass).
            log_dir: The directory to save logs and checkpoints.
            device: The device to run training on ('cpu' or 'cuda').
        """
        # Convert the dataclass config into the nested dictionary format
        # that the RSL-RL base runner expects.
        runner_internal_cfg = {
            "policy": asdict(train_cfg.policy),
            "algorithm": asdict(train_cfg.algorithm)
        }
        # Add all fields from RunnerConfig to the top level of the dict.
        for f in fields(train_cfg.runner):
            runner_internal_cfg[f.name] = getattr(train_cfg.runner, f.name)

        # Initialize the parent class with the constructed dictionary
        super().__init__(env, runner_internal_cfg, log_dir, device)

        # Store the original dataclass config for convenient access
        self.full_cfg = train_cfg

        # Set up video recording parameters from the config
        self.video_log_interval = self.full_cfg.runner.video_log_interval
        self.video_length = self.full_cfg.runner.video_length
        self.last_video_log_time = 0

    def log_video(self, it: int):
        """
        Records and logs a video of the policy interacting with the environment.

        Args:
            it (int): The current training iteration, used for the video filename.
        """
        if not self.writer or not hasattr(self.env, 'start_video_recording'):
            print("Video logging skipped: writer or recording methods not available.")
            return

        print("--- Recording video ---")
        video_name = f'iteration_{it}.mp4'
        video_path = os.path.join(self.log_dir, video_name)

        # Start recording through the environment's interface
        self.env.start_video_recording()

        # Run a short episode to generate the video frames
        obs, infos = self.env.get_observations()
        critic_obs = infos.get("observations", {}).get("critic", obs)
        for _ in range(self.video_length):
            with torch.no_grad():
                actions = self.alg.act(obs, critic_obs)
            obs, _, _, _, infos = self.env.step(actions)
            critic_obs = infos.get("observations", {}).get("critic", obs)

        # Stop recording and save the video file
        self.env.stop_video_recording(video_path)
        print(f"--- Video saved to {video_path} ---")

        # Log the saved video file to the appropriate logger (W&B or Tensorboard)
        try:
            if isinstance(self.writer, WandbSummaryWriter):
                wandb.log({"policy_rollout": wandb.Video(
                    video_path, fps=30, format="mp4")}, step=it)
                print("--- Logged video to W&B ---")
            else:  # Default to Tensorboard SummaryWriter
                with imageio.get_reader(video_path) as video_reader:
                    fps = video_reader.get_meta_data()['fps']
                    video_frames = np.array([frame for frame in video_reader])
                # Convert (T, H, W, C) to (N, T, C, H, W) for Tensorboard
                video_tensor = torch.from_numpy(
                    video_frames).permute(0, 3, 1, 2).unsqueeze(0)
                self.writer.add_video(
                    "policy_rollout", video_tensor, global_step=it, fps=fps)
                print("--- Logged video to Tensorboard ---")
        except Exception as e:
            print(f"Error logging video: {e}")

    def get_inference_policy(self, device=None):
        """Returns the policy in evaluation mode for inference."""
        self.alg.actor_critic.eval()
        return self.alg.actor_critic.act_inference

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        Overrides the main `learn` method to inject custom logging logic.

        This method follows the structure of the base RSL-RL `learn` method but adds:
        - Logging of individual reward components.
        - Logging of curriculum status.
        - Periodic calls to `log_video`.
        """
        # --- Initialization ---
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type == "wandb":
                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
            else:  # "tensorboard"
                self.writer = SummaryWriter(log_dir=self.log_dir)

        obs, infos = self.env.get_observations()
        critic_obs = infos.get("observations", {}).get("critic", obs)
        self.alg.actor_critic.train()

        # --- Logging Buffers ---
        # Standard buffers from base class
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device)

        # Buffers for decomposed reward components
        reward_component_names = [
            "distance", "success", "action_penalty", "joint_limit_penalty",
            "collision_penalty", "accel_penalty", "upright_bonus",
            "jerk_penalty", "joint_velocity_penalty", "ee_velocity_penalty"
        ]
        reward_component_buffers = {name: deque(
            maxlen=100) for name in reward_component_names}
        current_reward_component_sums = {
            name: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) for name in reward_component_names
        }

        tot_time = 0
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # --- Main Training Loop ---
        for it in range(start_iter, tot_iter):
            start = time.time()
            # --- Data Collection (Rollout) Phase ---
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, _, rews, dones, infos = self.env.step(actions)
                    critic_obs = infos.get(
                        "observations", {}).get("critic", obs)
                    self.alg.process_env_step(rews, dones, infos)

                    # Update reward and length sums
                    cur_reward_sum += rews
                    cur_episode_length += 1
                    for name in reward_component_names:
                        if f"reward_components/{name}" in infos:
                            current_reward_component_sums[
                                name] += infos[f"reward_components/{name}"]

                    # Handle finished episodes
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids].zero_()
                        cur_episode_length[new_ids].zero_()

                        # Move decomposed reward sums to their respective buffers
                        for name, sums in current_reward_component_sums.items():
                            reward_component_buffers[name].extend(
                                sums[new_ids][:, 0].cpu().numpy().tolist())
                            sums[new_ids].zero_()

                # Trigger video logging at the specified interval
                if self.video_log_interval > 0 and (it % self.video_log_interval == 0):
                    self.log_video(it)

                # Compute returns before the learning phase
                self.alg.compute_returns(critic_obs)

            collection_time = time.time() - start

            # --- Learning Phase ---
            learn_start = time.time()
            # Capture all five loss values from the update for logging
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
            learn_time = time.time() - learn_start
            self.current_learning_iteration = it

            # --- Logging Phase ---
            if self.log_dir is not None:
                # Retrieve episode infos from the environment for logging
                ep_infos = self.env.get_episode_infos()
                log_losses = (mean_value_loss, mean_surrogate_loss,
                              mean_entropy, mean_rnd_loss, mean_symmetry_loss)
                self._log_custom_metrics(it, infos, reward_component_buffers, rewbuffer, lenbuffer,
                                         log_losses, ep_infos, collection_time, learn_time)

            # --- Checkpointing ---
            if self.save_interval > 0 and (it % self.save_interval == 0):
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))

            tot_time += collection_time + learn_time

        # Save the final model
        self.save(os.path.join(self.log_dir,
                  f'model_{self.current_learning_iteration}.pt'))

    def _log_custom_metrics(self, it: int, infos: dict, reward_buffers: dict,
                            rewbuffer: deque, lenbuffer: deque,
                            losses: tuple, ep_infos: list,
                            collection_time: float, learn_time: float):
        """Logs detailed custom metrics to the writer."""
        if not self.writer:
            return

        # Log timings
        self.writer.add_scalar('Time/collection', collection_time, it)
        self.writer.add_scalar('Time/learn', learn_time, it)
        if len(lenbuffer) > 0:
            self.writer.add_scalar('Train/mean_reward',
                                   statistics.mean(rewbuffer), it)
            self.writer.add_scalar(
                'Train/mean_episode_length', statistics.mean(lenbuffer), it)

        mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = losses
        # Log main PPO losses, only if they are not None
        if mean_value_loss is not None:
            self.writer.add_scalar('Loss/value', mean_value_loss, it)
        if mean_surrogate_loss is not None:
            self.writer.add_scalar('Loss/surrogate', mean_surrogate_loss, it)
        if mean_entropy is not None:
            self.writer.add_scalar('Loss/entropy', mean_entropy, it)
        if mean_rnd_loss is not None:
            self.writer.add_scalar('Loss/rnd', mean_rnd_loss, it)
        if mean_symmetry_loss is not None:
            self.writer.add_scalar('Loss/symmetry', mean_symmetry_loss, it)

        # Log episode-specific info (like game scores, etc.)
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, it)

        # Log explained variance
        explained_var = 1 - torch.var(self.alg.storage.values -
                                      self.alg.storage.returns) / torch.var(self.alg.storage.returns)
        self.writer.add_scalar("Train/explained_variance",
                               explained_var.item(), it)

        # Log KL divergence if available
        if hasattr(self.alg.storage, 'kl'):
            self.writer.add_scalar("Train/kl_divergence",
                                   self.alg.storage.kl.mean().item(), it)

        # Log mean of each individual reward component
        for name, buffer in reward_buffers.items():
            if len(buffer) > 0:
                self.writer.add_scalar(
                    f"Rewards/mean_{name}", statistics.mean(buffer), it)

        # Log curriculum and debug info from the environment's `extras` dict
        for key, value in infos.items():
            if "curriculum/" in key:
                log_name = "Curriculum/" + key.split('/')[1]
                self.writer.add_scalar(log_name, value.item(), it)
            elif "debug/" in key:
                log_name = "Debug/" + key.split('/')[1]
                self.writer.add_scalar(log_name, value.mean().item(), it)


@dataclass
class EnvConfig:
    """Configuration for the Franka environment."""
    num_envs: int = 3072
    num_obs: int = 45
    num_privileged_obs: int = 0
    num_actions: int = 7
    max_episode_length: int = 200
    seed: int = 42

    # --- Control and Simulation ---
    control_mode: str = 'torque'  # 'velocity' or 'torque'
    dt: float = 0.005
    num_actions_history: int = 2

    # --- Base Reward Coefficients ---
    k_dist_reward: float = 0.0
    k_joint_limit_penalty: float = 0.0
    k_collision_penalty: float = 0.0
    success_reward_val: float = 300.0
    min_episode_length_for_success_metric: int = 10

    # --- Proximity-based velocity penalty ---
    proximity_vel_penalty_max_scale: float = 1.0
    proximity_vel_penalty_dist_threshold: float = 0.05

    # --- Modular Curriculum Configurations ---
    threshold_curriculum: CurriculumConfig = field(default_factory=dict)
    joint_velocity_penalty_curriculum: CurriculumConfig = field(
        default_factory=dict)
    ee_velocity_penalty_curriculum: CurriculumConfig = field(
        default_factory=dict)
    action_penalty_curriculum: CurriculumConfig = field(default_factory=dict)
    accel_penalty_curriculum: CurriculumConfig = field(default_factory=dict)
    jerk_penalty_curriculum: CurriculumConfig = field(default_factory=dict)
    upright_bonus_curriculum: CurriculumConfig = field(default_factory=dict)

    def __post_init__(self):
        """
        Dynamically sets reward coefficients and curricula based on the selected
        control mode after the object has been initialized.
        """
        if self.control_mode == 'velocity':
            self.k_dist_reward = 2.0
            self.k_joint_limit_penalty = 5.0
            self.k_collision_penalty = 20.0
            self.action_penalty_curriculum = CurriculumConfig(
                start_value=0.0005, end_value=0.0005, start_metric_val=0.0, end_metric_val=1.0)
            self.accel_penalty_curriculum = CurriculumConfig(
                start_value=0.0001, end_value=0.0001, start_metric_val=0.0, end_metric_val=1.0)
            self.jerk_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=0.0, start_metric_val=0.0, end_metric_val=1.0)
            self.joint_velocity_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=0.0, start_metric_val=0.0, end_metric_val=1.0)
            self.ee_velocity_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=0.0, start_metric_val=0.0, end_metric_val=1.0)
            self.upright_bonus_curriculum = CurriculumConfig(
                start_value=0.0, end_value=0.0, start_metric_val=0.0, end_metric_val=1.0)
            self.threshold_curriculum = CurriculumConfig(
                start_value=0.05, end_value=0.005, start_metric_val=0.7, end_metric_val=0.9)
        elif self.control_mode == 'torque':
            self.k_dist_reward = 2.5
            self.k_joint_limit_penalty = 5.0
            self.k_collision_penalty = 20.0
            self.action_penalty_curriculum = CurriculumConfig(
                start_value=1.0e-4, end_value=1.0e-3, start_metric_val=0.0, end_metric_val=0.2)
            self.accel_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=1.0e-6, start_metric_val=0.4, end_metric_val=0.6)
            self.jerk_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=1.0e-12, start_metric_val=0.7, end_metric_val=0.8)
            self.joint_velocity_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=0.5, start_metric_val=0.85, end_metric_val=0.95)
            self.ee_velocity_penalty_curriculum = CurriculumConfig(
                start_value=0.0, end_value=0.5, start_metric_val=0.85, end_metric_val=0.95)
            self.upright_bonus_curriculum = CurriculumConfig(
                start_value=1.0, end_value=0.0, start_metric_val=0.0, end_metric_val=0.4)
            self.threshold_curriculum = CurriculumConfig(
                start_value=0.05, end_value=0.005, start_metric_val=0.5, end_metric_val=0.84)

    # --- Task and Asset Config ---
    include_shelf: bool = False
    randomize_shelf_config: bool = True
    video_camera_pos: tuple = (1.8, -1.8, 2.0)
    video_camera_lookat: tuple = (0.3, 0.0, 0.5)
    video_camera_fov: float = 45
    video_res: tuple = (960, 640)
    franka_xml_path: str = 'xml/franka_emika_panda/panda.xml'


@dataclass
class PolicyConfig:
    """Configuration for the Actor-Critic policy network."""
    class_name: str = 'ActorCritic'
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 512, 256])
    activation: str = 'elu'
    init_noise_std: float = 1.0


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
    """Configuration for the training runner."""
    # --- General ---
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    max_iterations: int = 2501

    # --- RSL-RL Runner Parameters ---
    empirical_normalization: bool = True
    num_steps_per_env: int = 24
    logger: str = "wandb"  # 'wandb' or 'tensorboard'

    # --- Logging and Saving ---
    log_dir: str = "./training_logs/rsl_ppo_franka/"
    run_name: str = ""
    experiment_name: str = "trajectory-tracking-franka"
    save_interval: int = 100
    checkpoint_path: str = ""

    # --- Video Logging ---
    video_log_interval: int = 500
    video_length: int = 600

    # --- W&B Integration ---
    wandb: bool = True
    wandb_project: str = "franka-trajectory-tracking"
    wandb_entity: str = None  # Fetched automatically if None
    wandb_group: str = ""
    wandb_tags: list = field(default_factory=lambda: [
                             "rsl_rl", "franka", "reach", "trajectory-tracking"])


@dataclass
class TrainConfig:
    """Top-level configuration container that aggregates all other configs."""
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
