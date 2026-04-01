import gymnasium as gym
#  It gives us virtual env

from stable_baselines3 import A2C
# it gives us algo which act like brains

# Create a env
env=gym.make("LunarLander-v3",render_mode='human')

# Before stepping in the env we have to reset it first
env.reset()

# # We will see the example of a sample in the env we have made earlier
# print("Sample Action : ",env.action_space.sample())
# print("Observation : ",env.observation_space.shape)
# print("Sample observation",env.observation_space.sample())


# Agent
model=A2C('MlpPolicy',env,verbose=1)
model.learn(total_timesteps=10000)

episodes=10

for ep in range(episodes):
    obs=env.reset()
    done=False
    while not done:
        env.render()
        # show what environment currently looks like
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)


env.close()

