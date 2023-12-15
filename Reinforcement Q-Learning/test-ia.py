"""Training the agent"""
import gymnasium as gym

env = gym.make("Taxi-v3", render_mode="human")

env.reset()

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.P[328]