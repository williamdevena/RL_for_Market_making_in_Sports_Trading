from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, dict_env_params, verbose=0):
        super().__init__(verbose)
        self.dict_env_params = dict_env_params

    def _on_training_start(self):
        #print(self.model.policy, self.training_env.observation_space, self.training_env.action_space)
        hparam_dict = {
            #"algorithm": self.model.__class__.__name__,
            #"total timesteps": self.model._total_timesteps,
            'a_s': self.dict_env_params['a_s'],
            'b_s': self.dict_env_params['b_s'],
            'k': self.dict_env_params['k'],
            "learning rate": self.model.learning_rate,
            #"learning starts": self.model._num_timesteps_at_start,
            #"exploration factor": self.model.,
            #"gamma": self.model.gamma,
            #"policy": str(self.model.policy),
            "Obs. space": str(self.training_env.observation_space),
            #"Action space": str(self.training_env.action_space),
            #"fufa": 2.34
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_rew_mean": 0,
            #"rollout/ep_len_mean": 0,
            #"train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )


    def _on_step(self):
        return True
