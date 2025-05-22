import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add current directory to path to allow import of train_agent
# This is often needed when running tests from a subdirectory or with certain test runners
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that train_agent.py imports and uses, especially heavy ones like stable_baselines3
# and the custom environment/extractor if they have heavy dependencies (like Genesis)

# Mock FrankaShelfEnv and CustomCNNFeatureExtractor before they are imported by train_agent
# This is crucial if their __init__ methods have side effects or dependencies we want to avoid in this test.
mock_env = MagicMock(spec_set=['observation_space', 'action_space', 'close'])
mock_env.observation_space = MagicMock()  # Further configure if needed by PPO
mock_env.action_space = MagicMock()

mock_extractor = MagicMock()

# We need to ensure that when train_agent imports these, it gets our mocks.
# This is typically done by adding the mocks to sys.modules BEFORE the import.
# However, for a script like train_agent, it's often easier to patch its internal imports.

# Patching at the point of use within train_agent.py


@patch('train_agent.FrankaShelfEnv', new=mock_env)
@patch('train_agent.CustomCNNFeatureExtractor', new=mock_extractor)
@patch('train_agent.PPO')  # Mock the PPO class itself
@patch('train_agent.CheckpointCallback')
# Mock os.makedirs to avoid filesystem operations
@patch('train_agent.os.makedirs')
# Mock check_env as it can be slow/problematic
@patch('train_agent.check_env', MagicMock())
class TestTrainAgentScript(unittest.TestCase):
    """
    Unit/Smoke tests for the train_agent.py script.
    Focuses on ensuring the script can be invoked and initializes components
    without crashing, rather than testing the full RL training loop.
    """

    def test_train_franka_agent_script_runs(self, mock_makedirs, mock_checkpoint_callback, mock_ppo_class):
        """
        Smoke test to ensure the main training function can be called
        and PPO model initialization is attempted with mocked dependencies.
        """
        # Mock the PPO model's learn and save methods
        mock_ppo_instance = MagicMock()
        mock_ppo_instance.learn = MagicMock()
        mock_ppo_instance.save = MagicMock()
        # PPO(...) returns our mock
        mock_ppo_class.return_value = mock_ppo_instance

        # Mock the FrankaShelfEnv constructor to return our mock_env instance
        # This is now handled by the class-level patch `train_agent.FrankaShelfEnv`

        # Dynamically import train_agent *after* patches are set up, if it's a script
        # Or, if it's a module with functions, call the function directly.
        # For this structure, we assume train_agent.py can be imported and has train_franka_agent
        try:
            # If train_agent.py is structured as a script with if __name__ == '__main__':
            # directly importing and calling its main function is one way.
            # Here, we assume train_franka_agent is a callable function.
            import train_agent
            train_agent.train_franka_agent()  # Call the main function
        except Exception as e:
            self.fail(
                f"train_franka_agent script failed to run with mocks: {e}")

        # Assert that PPO was initialized (i.e., PPO class was called)
        mock_ppo_class.assert_called_once()

        # Assert that the environment was instantiated (FrankaShelfEnv was called)
        # The class-level patch means mock_env is FrankaShelfEnv.
        # We need to check if it was called.
        # If FrankaShelfEnv is directly patched to mock_env (the instance),
        # then we check if methods on mock_env were called if PPO interacts with it.
        # If FrankaShelfEnv is patched to a mock *class*, then assert that class was called.
        # In our case, `@patch('train_agent.FrankaShelfEnv', new=mock_env)` replaces the class
        # with an instance, so PPO will receive this instance.
        # Let's verify PPO was called with our mock_env.
        args, kwargs = mock_ppo_class.call_args
        self.assertIs(kwargs['env'], mock_env)  # Check if PPO got our env mock

        # Assert that model.learn was called
        mock_ppo_instance.learn.assert_called_once()

        # Assert that model.save was called (in the finally block)
        mock_ppo_instance.save.assert_called_once()

        # Assert that env.close was called
        mock_env.close.assert_called_once()

        # Assert that os.makedirs was called for the log directory
        mock_makedirs.assert_called()


if __name__ == '__main__':
    # This is a bit tricky because train_agent.py might also have an if __name__ == '__main__'
    # For testing, it's better if train_agent.py's main logic is in a function.
    # Assuming train_agent.py is importable and its main function is train_franka_agent.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
