# Created by Patrick Kao
import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy


class LargeEvalCallback(BaseCallback):
    def __init__(self, n_steps=70000, n_eval_episodes=2000, verbose=0):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.n_eval_episodes = n_eval_episodes
        self.last_time_trigger = 0

    def _on_rollout_start(self) -> None:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            value, _ = evaluate_policy(self.model, self.training_env,
                                    n_eval_episodes=self.n_eval_episodes, )
            summary = tf.Summary(value=[tf.Summary.Value(tag='large_eval_performance', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
