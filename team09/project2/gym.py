import gym
from qlearning import QLearningAgent
from qlearning import train
import numpy as np

env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
alpha = 0.5
gamma = 0.99
epsilon = 0.1
agent = QLearningAgent(n_states, n_actions, alpha, gamma, epsilon)
n_episodes = 1000
rewards = train(env, agent, n_episodes)

print('Average reward:', np.mean(rewards))