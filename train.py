import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from vgdl.util.humanplay.play_vgdl import register_vgdl_env

os.environ['KMP_DUPLICATE_LIB_OK']='True'
domain_file = 'vgdl/games/aliens.txt'
level_file = 'vgdl/games/aliens_lvl0.txt'
model = 'MlpPolicy'

def train(model, domain_file, level_file, observer = 'image', blocksize = 5, steps = 10000):
	'''Training function, includes domain file with VGDL rules and designed level to render environment from.'''
	env_name = register_vgdl_env(domain_file, level_file, observer, blocksize)
	env = gym.make(env_name)
	# Render once to initialize the viewer
	env.render(mode='human')

	model = PPO(model, env, verbose=1).learn(steps)