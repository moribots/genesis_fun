import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys

# --- Mock external dependencies before they are imported by the module under test ---

# Mock Genesis (gs)
mock_gs = MagicMock()
mock_gs.gpu = "gpu_backend_mock"
mock_gs.cpu = "cpu_backend_mock"
mock_gs._is_initialized = False  # Simulate not initialized at first
sys.modules['genesis'] = mock_gs  # Replace the actual genesis module

# Mock FrankaShelfEnv class and instance
mock_franka_shelf_env_class = MagicMock(name="FrankaShelfEnvClass")
mock_franka_shelf_env_instance = MagicMock(name="FrankaShelfEnvInstance")
mock_franka_shelf_env_instance.observation_space = MagicMock()
mock_franka_shelf_env_instance.action_space = MagicMock()
mock_franka_shelf_env_class.return_value = mock_franka_shelf_env_instance

# Mock PPO class and instance
mock_ppo_class = MagicMock(name="PPOClass")
mock_ppo_instance = MagicMock(name="PPOInstance")
mock_ppo_instance.learn = MagicMock()
mock_ppo_instance.save = MagicMock()
mock_ppo_class.return_value = mock_ppo_instance

# Mock VecEnv classes and a generic VecEnv instance
mock_dummy_vec_env_class = MagicMock(name="DummyVecEnvClass")
mock_subproc_vec_env_class = MagicMock(name="SubprocVecEnvClass")
mock_vec_env_instance = MagicMock(name="VecEnvInstance")
mock_vec_env_instance.observation_space = MagicMock()
mock_vec_env_instance.action_space = MagicMock()
mock_vec_env_instance.close = MagicMock()
mock_vec_env_instance.save = MagicMock()  # Add save mock for VecNormalize
mock_dummy_vec_env_class.return_value = mock_vec_env_instance
mock_subproc_vec_env_class.return_value = mock_vec_env_instance

# Mock VecNormalize (it's also a VecEnv, so needs similar mocks)
mock_vec_normalize_class = MagicMock(name="VecNormalizeClass")
# VecNormalize instance should behave like a VecEnv and have a save method
mock_vec_normalize_instance = MagicMock(name="VecNormalizeInstance")
mock_vec_normalize_instance.observation_space = MagicMock()
mock_vec_normalize_instance.action_space = MagicMock()
mock_vec_normalize_instance.close = MagicMock()
mock_vec_normalize_instance.save = MagicMock()
mock_vec_normalize_class.return_value = mock_vec_normalize_instance


# Mock Callbacks
mock_checkpoint_callback_class = MagicMock(name="CheckpointCallbackClass")
mock_eval_callback_class = MagicMock(name="EvalCallbackClass")

# Mock utility functions
mock_make_vec_env = MagicMock(name="make_vec_env_func")
# make_vec_env will return a plain VecEnv, which is then wrapped by VecNormalize
# So, make_vec_env should return a mock that can be wrapped.
# The mock_vec_env_instance will serve for the direct output of make_vec_env.
# And mock_vec_normalize_instance will be the result of wrapping with VecNormalize.
mock_make_vec_env.return_value = mock_vec_env_instance

mock_set_random_seed = MagicMock(name="set_random_seed_func")

# Mock for os.makedirs and sys.exit
mock_os_makedirs = MagicMock(name="makedirs_mock")
mock_sys_exit = MagicMock(name="sys_exit_mock")

# Mock for torch.cuda.is_available
mock_torch_cuda_is_available = MagicMock()

# --- Mocks for W&B ---
mock_wandb_login = MagicMock(name="wandb_login_mock")
mock_wandb_run_instance = MagicMock(name="WandbRunInstance")
mock_wandb_run_instance.url = "mock_wandb_run_url"
mock_wandb_run_instance.id = "mock_run_id"
mock_wandb_run_instance.finish = MagicMock()
mock_wandb_run_instance.log_artifact = MagicMock()

mock_wandb_init = MagicMock(name="wandb_init_mock",
                            return_value=mock_wandb_run_instance)
mock_wandb_artifact_class = MagicMock(name="WandbArtifactClass")
mock_wandb_callback_class = MagicMock(name="WandbCallbackClass")
mock_getpass = MagicMock(name="getpass_mock", return_value="mock_api_key")


# Patching at the point of use within train_agent.py
@patch('train_agent.FrankaShelfEnv', new=mock_franka_shelf_env_class)
@patch('train_agent.PPO', new=mock_ppo_class)
@patch('train_agent.DummyVecEnv', new=mock_dummy_vec_env_class)
@patch('train_agent.SubprocVecEnv', new=mock_subproc_vec_env_class)
# Patches the VecNormalize class
@patch('train_agent.VecNormalize', new=mock_vec_normalize_class)
@patch('train_agent.CheckpointCallback', new=mock_checkpoint_callback_class)
@patch('train_agent.EvalCallback', new=mock_eval_callback_class)
@patch('train_agent.make_vec_env', new=mock_make_vec_env)
@patch('train_agent.set_random_seed', new=mock_set_random_seed)
@patch('train_agent.os.makedirs', new=mock_os_makedirs)
@patch('train_agent.sys.exit', new=mock_sys_exit)
@patch('train_agent.torch.cuda.is_available', new=mock_torch_cuda_is_available)
@patch('train_agent.wandb.login', new=mock_wandb_login)
@patch('train_agent.wandb.init', new=mock_wandb_init)
@patch('train_agent.wandb.Artifact', new=mock_wandb_artifact_class)
# Assumes 'from wandb.integration.sb3 import WandbCallback'
@patch('train_agent.WandbCallback', new=mock_wandb_callback_class)
@patch('train_agent.getpass.getpass', new=mock_getpass)
class TestTrainAgentScript(unittest.TestCase):
    """
    Unit/Smoke tests for the train_agent.py script.
    """

    def setUp(self):
        # Reset mocks before each test
        mock_gs.reset_mock()
        mock_gs._is_initialized = False
        mock_gs.init.side_effect = None

        mock_franka_shelf_env_class.reset_mock()
        mock_franka_shelf_env_instance.reset_mock()
        mock_ppo_class.reset_mock()
        mock_ppo_instance.reset_mock()
        mock_dummy_vec_env_class.reset_mock()
        mock_subproc_vec_env_class.reset_mock()
        mock_vec_env_instance.reset_mock()  # This is returned by make_vec_env
        mock_vec_normalize_class.reset_mock()  # This is the class
        # This is the instance returned by VecNormalize()
        mock_vec_normalize_instance.reset_mock()
        # Ensure VecNormalize class returns our specific mock instance
        mock_vec_normalize_class.return_value = mock_vec_normalize_instance

        mock_checkpoint_callback_class.reset_mock()
        mock_eval_callback_class.reset_mock()
        mock_make_vec_env.reset_mock()
        # make_vec_env returns mock_vec_env_instance, which is then wrapped by VecNormalize
        mock_make_vec_env.return_value = mock_vec_env_instance

        mock_set_random_seed.reset_mock()
        mock_os_makedirs.reset_mock()
        mock_sys_exit.reset_mock()
        mock_sys_exit.side_effect = None
        mock_torch_cuda_is_available.reset_mock()

        # Reset W&B mocks
        mock_wandb_login.reset_mock()
        mock_wandb_init.reset_mock()
        # Ensure it returns the run instance
        mock_wandb_init.return_value = mock_wandb_run_instance
        mock_wandb_run_instance.reset_mock()
        mock_wandb_artifact_class.reset_mock()
        mock_wandb_callback_class.reset_mock()
        mock_getpass.reset_mock()
        mock_getpass.return_value = "mock_api_key"  # Default return for getpass

    def test_train_franka_agent_script_runs_main_logic(self):
        """
        Smoke test for the main training function.
        """
        # Import train_agent *after* all patches are set up.
        import train_agent

        mock_torch_cuda_is_available.return_value = False  # Default to CPU

        test_total_timesteps = 100
        test_num_envs = 2
        test_seed = 123
        test_log_dir = "./test_logs/"
        test_model_save_path = "./test_model_save/model"
        test_wandb_project_name = "TestProject"

        train_agent.train_franka_agent(
            total_timesteps=test_total_timesteps,
            num_envs=test_num_envs,
            seed=test_seed,
            log_dir=test_log_dir,
            model_save_path=test_model_save_path,
            wandb_project_name=test_wandb_project_name,  # Ensure W&B path is tested
            wandb_entity=None,
            use_cnn=False  # Test without CNN for simplicity here, or add CNN mock
        )

        mock_gs.init.assert_called_once_with(backend=mock_gs.cpu)
        mock_wandb_login.assert_called_once()
        mock_wandb_init.assert_called_once()
        # Check some wandb.init args - be more specific if needed
        self.assertEqual(
            mock_wandb_init.call_args[1]['project'], test_wandb_project_name)

        mock_os_makedirs.assert_any_call(test_log_dir, exist_ok=True)
        # Check if the directory for model_save_path is created
        if os.path.dirname(test_model_save_path):
            mock_os_makedirs.assert_any_call(
                os.path.dirname(test_model_save_path), exist_ok=True)

        mock_set_random_seed.assert_called_once_with(test_seed)

        # make_vec_env is called for train_env and eval_env
        self.assertEqual(mock_make_vec_env.call_count, 2)
        # VecNormalize is called for train_env and eval_env
        self.assertEqual(mock_vec_normalize_class.call_count, 2)
        # The first call to VecNormalize wraps the first make_vec_env result
        # The second call to VecNormalize wraps the second make_vec_env result
        mock_vec_normalize_class.assert_any_call(
            mock_vec_env_instance, norm_obs=True, norm_reward=True, gamma=unittest.mock.ANY)
        mock_vec_normalize_class.assert_any_call(
            mock_vec_env_instance, training=False, norm_obs=True, norm_reward=False, gamma=unittest.mock.ANY)

        # PPO is initialized with the VecNormalize instance for the training env
        mock_ppo_class.assert_called_once_with(
            "MultiInputPolicy",
            # This is the instance returned by VecNormalize wrapping train_env
            mock_vec_normalize_instance,
            policy_kwargs=None,  # Since use_cnn=False
            verbose=1,
            tensorboard_log=test_log_dir,  # or None if run is active and wandb syncs
            seed=test_seed,
            gamma=unittest.mock.ANY,
            gae_lambda=unittest.mock.ANY,
            n_steps=unittest.mock.ANY,
            batch_size=unittest.mock.ANY,
            n_epochs=unittest.mock.ANY,
            learning_rate=unittest.mock.ANY,
            clip_range=unittest.mock.ANY,
            ent_coef=unittest.mock.ANY,
            vf_coef=unittest.mock.ANY,
            max_grad_norm=unittest.mock.ANY
        )

        # Check callbacks
        self.assertTrue(mock_checkpoint_callback_class.called)
        self.assertTrue(mock_eval_callback_class.called)
        self.assertTrue(mock_wandb_callback_class.called)  # Check W&B callback

        # Check learn call
        mock_ppo_instance.learn.assert_called_once()
        # Verify the callbacks list passed to model.learn
        learn_callbacks = mock_ppo_instance.learn.call_args[1]['callback']
        self.assertIn(
            mock_checkpoint_callback_class.return_value, learn_callbacks)
        self.assertIn(mock_eval_callback_class.return_value, learn_callbacks)
        self.assertIn(mock_wandb_callback_class.return_value, learn_callbacks)

        # Check model save and VecNormalize save
        expected_final_model_name = os.path.basename(
            test_model_save_path.rstrip('/\\'))
        full_final_model_path = os.path.join(os.path.dirname(
            test_model_save_path) or test_log_dir, f"{expected_final_model_name}_final.zip")
        mock_ppo_instance.save.assert_called_once_with(full_final_model_path)

        # The training 'env' (which is mock_vec_normalize_instance) should be saved
        vec_normalize_stats_path = os.path.join(os.path.dirname(
            test_model_save_path) or test_log_dir, f"{expected_final_model_name}_vecnormalize.pkl")
        mock_vec_normalize_instance.save.assert_any_call(
            vec_normalize_stats_path)

        # Check envs are closed
        # mock_vec_normalize_instance is used for both train and eval env wrappers in this setup
        # if they were distinct mocks, we'd check each.
        # The `env.close()` and `eval_env.close()` in train_agent will call close on the
        # VecNormalize wrappers (which are both `mock_vec_normalize_instance` in the test if not careful).
        # Let's refine mock setup for VecNormalize for train and eval separately if needed.
        # For now, VecNormalize class is called twice, and its return (mock_vec_normalize_instance) has close called.
        # If VecNormalize class returns a new MagicMock each time:
        # train_env_wrapper_mock = MagicMock(); eval_env_wrapper_mock = MagicMock()
        # mock_vec_normalize_class.side_effect = [train_env_wrapper_mock, eval_env_wrapper_mock]
        # Then check train_env_wrapper_mock.close() and eval_env_wrapper_mock.close()
        # For simplicity, if mock_vec_normalize_class always returns the same global mock_vec_normalize_instance:
        self.assertGreaterEqual(
            mock_vec_normalize_instance.close.call_count, 2)

        # Check W&B artifacts and finish
        mock_wandb_artifact_class.assert_any_call(
            unittest.mock.ANY, type='model')
        mock_wandb_artifact_class.assert_any_call(
            unittest.mock.ANY, type='dataset')
        self.assertGreaterEqual(
            mock_wandb_run_instance.log_artifact.call_count, 2)
        mock_wandb_run_instance.finish.assert_called_once()

        if hasattr(mock_gs, 'shutdown') and callable(mock_gs.shutdown):
            mock_gs.shutdown.assert_called_once()

    def test_genesis_init_gpu(self):
        """Test Genesis initialization with GPU if available."""
        import train_agent
        mock_torch_cuda_is_available.return_value = True  # Force GPU
        train_agent.train_franka_agent(
            total_timesteps=10, num_envs=1, wandb_project_name="TestGPU")  # W&B active
        mock_gs.init.assert_called_once_with(backend=mock_gs.gpu)
        mock_wandb_init.assert_called_once()  # Ensure W&B init is still called

    def test_genesis_init_failure(self):
        """Test behavior when Genesis initialization fails."""
        import train_agent
        mock_gs.init.side_effect = Exception("Genesis Boom!")
        mock_sys_exit.side_effect = lambda code: (
            _ for _ in ()).throw(SystemExit(code))

        with self.assertRaises(SystemExit) as cm:
            # Pass wandb_project_name=None to avoid W&B init attempt after expected exit
            train_agent.train_franka_agent(
                total_timesteps=10, num_envs=1, wandb_project_name=None)

        mock_sys_exit.assert_called_once_with(1)
        self.assertEqual(cm.exception.code, 1)
        # W&B init should not be called if Genesis fails first
        mock_wandb_init.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
