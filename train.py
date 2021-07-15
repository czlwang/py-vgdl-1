import sys
import os
import gym
import yaml #pip install pyyaml
import numpy as np
import configargparse

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from vgdl.util.humanplay.play_vgdl import register_vgdl_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["SDL_VIDEODRIVER"] = "dummy"

#parse command line arguments
parser = configargparse.ArgumentParser()
parser.add_argument("-c", "--config", is_config_file=True)
parser.add_argument('--domain_file', type=str, help='path to the VGDL game file')
parser.add_argument('--level_file', type=str, help='path to the VGDL game level file')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--observer', type=str, help='observation type')
parser.add_argument('--save_dir', type=str, help='path to save model at')
parser.add_argument('--log_dir', type=str, help='path to save logs')
parser.add_argument('--steps', type=int, help='number of training steps')
parser.add_argument('--algo', type=str, help='training algorithm, ex: PPO, DQN')
parser.add_argument('--name', type=str, help='name of the run')
args = parser.parse_args()

def train(args):
    '''Training function, includes domain file with VGDL rules and 
       designed level to render environment from.'''

    domain_file = args.domain_file
    level_file = args.level_file
    env_name = register_vgdl_env(domain_file, level_file, 
                                 args.observer, blocksize=5)

    env = gym.make(env_name)
    save_dir = args.save_dir
    env = Monitor(env, save_dir)

    if args.observer=='image':
        env.render(mode='human')
        #env.close()

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_dir=save_dir)
    
    algos = {"PPO":PPO, "DQN":DQN}
    algo = algos[args.algo]
    log_dir = args.log_dir
    model = algo(args.model, env, verbose=1, tensorboard_log=log_dir,
                  create_eval_env=True) 

    name = args.name
    if args.name is None:
       name = algo

    model.learn(args.steps, callback=callback, eval_freq=1000, 
                tb_log_name=name, n_eval_episodes=5, 
                eval_log_path="./logs/")
    
    policy = model.policy
    env = model.get_env()
    mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    with open("mean_rewards.txt", "a") as f:
        f.write(f"{domain_file} mean_reward={mean_reward:.2f} +/- {std_reward}")
    return model

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
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
log_dir = args.log_dir
os.makedirs(log_dir, exist_ok=True)

train(args)
