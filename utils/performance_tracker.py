import numpy as np
from typing import Dict, List

class PerformanceTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = []
        self.lengths = []
        self.steps = []
        
    def update(self, reward, length, steps):
        """Update performance metrics."""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.steps.append(steps)
        
        # Keep only the last window_size entries
        if len(self.rewards) > self.window_size:
            self.rewards.pop(0)
            self.lengths.pop(0)
            self.steps.pop(0)
    
    def get_average_reward(self):
        """Get average reward over the window."""
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    def get_average_length(self):
        """Get average snake length over the window."""
        if not self.lengths:
            return 0.0
        return sum(self.lengths) / len(self.lengths)
    
    def get_average_steps(self):
        """Get average number of steps over the window."""
        if not self.steps:
            return 0.0
        return sum(self.steps) / len(self.steps)
    
    def get_stats(self):
        """Get all performance statistics."""
        return {
            'avg_reward': self.get_average_reward(),
            'avg_length': self.get_average_length(),
            'avg_steps': self.get_average_steps(),
            'max_length': max(self.lengths) if self.lengths else 0,
            'max_steps': max(self.steps) if self.steps else 0
        } 