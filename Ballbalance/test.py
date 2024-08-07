import os
import argparse
from datetime import datetime


# from gail_airl_ppo.env import make_env
# from isaac_gym_env import paramAnt
from gail_airl_ppo.algo import SAC
from gail_airl_ppo.trainer import DR_Trainer
import torch


x = 1
y = [x]
print(y)