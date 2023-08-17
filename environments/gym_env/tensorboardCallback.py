from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        #value = np.random.random()

        final_pnl = self.training_env.envs[0].env.pnl
        self.logger.record("Final PnL", final_pnl)
        #print("callback")

        #print(self.training_env.__dict__.keys())
        # print(self.training_env.envs[0].env.__dict__.keys())

        return True

    def _on_rollout_end(self) -> None:
        #return super()._on_rollout_end()
        #print("rollout end")

        return True