# example.py
import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    breakpoint()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()