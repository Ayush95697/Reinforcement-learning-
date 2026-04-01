# Saving , loading and tracking the model


import gymnasium as gym
import os

from stable_baselines3 import A2C,PPO

# We are creating path files to save the logs and info of models
models_dir="models/PPO"
logs_dir="logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

env=gym.make("LunarLander-v3",render_mode='human')
env.reset()

# Agent
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=logs_dir)
TIMESTEPS=10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
# This part is to used the learned model to see how it performs in the env
# episodes=10
#
# for ep in range(episodes):
#     obs=env.reset()
#     done=False
#     while not done:
#         env.render()
#         # show what environment currently looks like
#         action, _ = model.predict(obs)
#         obs, reward, terminated, truncated, info = env.step(action)

env.close()

