import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Deque
from collections import deque
import random
import json
from environment.board import Direction

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class QLearningAgent:
    def __init__(self, state_size, action_size, hidden_size=512):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Q-Network
        self.q_network = QNetwork(state_size, hidden_size, action_size)
        self.target_network = QNetwork(state_size, hidden_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Hyperparameters
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.memory_size = 100000
        self.batch_size = 64
        
        # Initialize memory
        self.memory = deque(maxlen=self.memory_size)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss()
        
        # Initialize step counter
        self.steps = 0
        
    def get_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state)
                self.q_network.train()
                return torch.argmax(q_values).item()
        return random.randrange(self.action_size)
    
    def train(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Start training only when we have enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([s for s, _, _, _, _ in batch])
        actions = torch.LongTensor([a for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_states = torch.FloatTensor([ns for _, _, _, ns, _ in batch])
        dones = torch.FloatTensor([d for _, _, _, _, d in batch])
        
        # Get current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str) -> None:
        """Save the model state and parameters."""
        model_state = {
            'network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'hidden_size': self.hidden_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'target_update': self.target_update,
                'memory_size': self.memory_size,
                'batch_size': self.batch_size
            }
        }
        torch.save(model_state, filepath)

    def load_model(self, filepath: str) -> None:
        """Load the model state and parameters."""
        model_state = torch.load(filepath)
        
        # Load hyperparameters
        hyperparameters = model_state['hyperparameters']
        self.state_size = hyperparameters['state_size']
        self.action_size = hyperparameters['action_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.target_update = hyperparameters['target_update']
        self.memory_size = hyperparameters['memory_size']
        self.batch_size = hyperparameters['batch_size']
        
        # Load network states
        self.q_network.load_state_dict(model_state['network_state'])
        self.target_network.load_state_dict(model_state['target_network_state'])
        self.optimizer.load_state_dict(model_state['optimizer_state'])
        self.epsilon = model_state['epsilon']

    def train_curriculum(self, num_sessions: int):
        """Progressive training with increasing difficulty."""
        stages = [
            {'state_size': 5, 'sessions': num_sessions // 4},
            {'state_size': 7, 'sessions': num_sessions // 4},
            {'state_size': 10, 'sessions': num_sessions // 2}
        ]
        
        for stage in stages:
            self.state_size = stage['state_size']
            self.train_sessions(stage['sessions']) 