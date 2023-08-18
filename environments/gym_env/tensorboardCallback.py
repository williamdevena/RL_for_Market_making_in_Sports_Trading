from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "total timesteps": self.model._total_timesteps,
            "learning rate": self.model.learning_rate,
            #"learning starts": self.model._num_timesteps_at_start,
            #"exploration factor": self.model.,
            "gamma": self.model.gamma,
            "policy": self.model.policy,
            "Obs. space": self.training_env.observation_space,
            "Action space": self.training_env.action_space,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

    # def _on_step(self) -> bool:
    #     # Log scalar value (here a random variable)
    #     #value = np.random.random()

    #     final_pnl = self.training_env.envs[0].env.pnl
    #     self.logger.record("Final PnL", final_pnl)
    #     #print("callback")

    #     #print(self.training_env.__dict__.keys())
    #     # print(self.training_env.envs[0].env.__dict__.keys())

    #     return True

    # def _on_rollout_end(self) -> None:
    #     #return super()._on_rollout_end()
    #     #print("rollout end")

    #     return True
