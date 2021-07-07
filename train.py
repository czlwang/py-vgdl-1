import sys
import os
import gym
import yaml #pip install pyyaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from vgdl.util.humanplay.play_vgdl import register_vgdl_env

os.environ['KMP_DUPLICATE_LIB_OK']='True'

arg_file = sys.argv[1]
with open(arg_file, 'r') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

def train(model, domain_file, level_file, 
          observer = 'image', blocksize = 5, steps = 10000):
    '''Training function, includes domain file with VGDL rules and 
       designed level to render environment from.'''

    env_name = register_vgdl_env(domain_file, level_file, 
                                 observer, blocksize)
    env = gym.make(env_name)
    if observer=='image':
        env.render(mode='human')
        env.close()

    model = PPO(model, env, verbose=1).learn(steps)


train(cfg["model"], cfg["domain_file"], 
      cfg["level_file"], observer=cfg["observer"],
      steps=cfg["steps"])
