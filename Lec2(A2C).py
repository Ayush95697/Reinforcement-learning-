import gymnasium as gym
import os
from stable_baselines3 import A2C

models_dir="models/A2C"
logs_dir="logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

env=gym.make("LunarLander-v3")

model=A2C("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir)

TIMESTEPS=10000

for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

# To check the logs in tensorboard, run the following command in terminal:
# tensorboard --logdir logs