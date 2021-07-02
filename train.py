import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from vgdl.util.humanplay.play_vgdl import register_vgdl_env

domain_file = 'vgdl/games/aliens.txt'
level_file = 'vgdl/games/aliens_lvl0.txt'
env_name = register_vgdl_env(domain_file, level_file, 'image', 5)
env = gym.make(env_name)
# Render once to initialize the viewer
env.render(mode='human')

model = PPO('MlpPolicy', env, verbose=1).learn(10000)
