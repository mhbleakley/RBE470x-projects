import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
        self.Q = np.zeros((n_states, n_actions))  # Q-values initialized to 0
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:  # explore
            return np.random.randint(self.Q.shape[1])
        else:  # exploit
            return np.argmax(self.Q[state, :])
        
    def update_Q(self, state, action, reward, next_state):
        max_Q = np.max(self.Q[next_state, :])
        self.Q[state, action] += self.alpha * (reward + self.gamma * max_Q - self.Q[state, action])

def run_episode(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_Q(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    
    return total_reward

def train(env, agent, n_episodes):
    rewards = []
    
    for i in range(n_episodes):
        total_reward = run_episode(env, agent)
        rewards.append(total_reward)
        
    return rewards