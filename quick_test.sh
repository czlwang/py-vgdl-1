#!/bin/bash

  python3 train.py --domain_file='vgdl/games/survivezombies.txt' --level_file='vgdl/games/survivezombies_lvl0.txt' --log_dir='runs' --model='MlpPolicy' --observer='features' --save_dir='output' --steps=100000
  python3 train.py --domain_file='vgdl/games/survivezombies_v1.txt' --level_file='vgdl/games/survivezombies_lvl0.txt' --log_dir='runs' --model='MlpPolicy' --observer='features' --save_dir='output' --steps=100000
  python3 train.py --domain_file='vgdl/games/survivezombies_v2.txt' --level_file='vgdl/games/survivezombies_lvl0.txt' --log_dir='runs' --model='MlpPolicy' --observer='features' --save_dir='output' --steps=100000
