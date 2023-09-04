from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class TensorboardCallback(BaseCallback):
    """
    Custom callback for Tensorboard integration during RL agent training.

    This callback logs additional environment parameters and model hyperparameters
    to Tensorboard.

    Attributes:
        dict_env_params (dict): A dictionary of environment parameters.
        verbose (int): Verbosity level.

    """

    def __init__(self, dict_env_params, verbose=0):
        """
        Initializes the TensorboardCallback object.

        Args:
            dict_env_params (dict): A dictionary containing key-value pairs of environment parameters.
            verbose (int, optional): Verbosity level. Defaults to 0.

        """
        super().__init__(verbose)
        self.dict_env_params = dict_env_params

    def _on_training_start(self):
        """
        Logs hyperparameters to Tensorboard at the start of training.

        This method is automatically called when training starts.
        """
        hparam_dict = {
            'model_save_path': self.dict_env_params['model_save_path'],
            'a_s': self.dict_env_params['a_s'],
            'b_s': self.dict_env_params['b_s'],
            'k': self.dict_env_params['k'],
            "learning rate": self.model.learning_rate,
            "Obs. space": str(self.training_env.observation_space),
        }
        metric_dict = {
            "rollout/ep_rew_mean": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self):
        """Method to keep the callback API consistent.

        This method is automatically called at each step of training.

        Returns:
            bool: Always returns True to continue training.
        """
        return True


    def on_rollout_end(self):
        """
        Logs environment parameters at the end of each rollout. It logs
        the three environment parameters: 'k', 'a_s' and 'b_s' (useful
        when the SportsTradingEnvironment is set with 'mode'='random').

        This method is automatically called at the end of each rollout during training.
        """
        self.logger.record("rollout/k", self.training_env.envs[0].env.k)
        self.logger.record("rollout/a_s", self.training_env.envs[0].env.a_s)
        self.logger.record("rollout/b_s", self.training_env.envs[0].env.b_s)



