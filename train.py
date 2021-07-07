import sys
import os
import gym
import yaml #pip install pyyaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from vgdl.util.humanplay.play_vgdl import register_vgdl_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

os.environ['KMP_DUPLICATE_LIB_OK']='True'

arg_file = sys.argv[1]
with open(arg_file, 'r') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

def train(model, domain_file, level_file, observer='image', 
          blocksize = 5, steps = 10000, log_dir=None, save_dir=None):
    '''Training function, includes domain file with VGDL rules and 
       designed level to render environment from.'''

    env_name = register_vgdl_env(domain_file, level_file, 
                                 observer, blocksize)

    env = gym.make(env_name)
    env = Monitor(env, save_dir)

    if observer=='image':
        env.render(mode='human')
        #env.close()

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_dir=save_dir)
    model = PPO(model, env, verbose=1, tensorboard_log=log_dir)
    model.learn(steps, callback=callback)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    From: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html

    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param save_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, save_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.save_path = os.path.join(save_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.save_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log and save dirs
save_dir = cfg["save_dir"]
os.makedirs(save_dir, exist_ok=True)
log_dir = cfg["log_dir"]
os.makedirs(log_dir, exist_ok=True)

train(cfg["model"], cfg["domain_file"], 
      cfg["level_file"], observer=cfg["observer"],
      steps=cfg["steps"], log_dir=log_dir,
      save_dir=save_dir)
