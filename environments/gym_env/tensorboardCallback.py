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
            'model_save_path': self.dict_env_params['model_save_path'],
            'a_s': self.dict_env_params['a_s'],
            'b_s': self.dict_env_params['b_s'],
            'k': self.dict_env_params['k'],
            "learning rate": self.model.learning_rate,
            "Obs. space": str(self.training_env.observation_space),
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


    def on_rollout_end(self):
        self.logger.record("rollout/k", self.training_env.envs[0].env.k)
        self.logger.record("rollout/a_s", self.training_env.envs[0].env.a_s)
        self.logger.record("rollout/b_s", self.training_env.envs[0].env.b_s)
        #print(self.training_env.__dict__)
        #print(self.training_env.envs[0].env.k)



