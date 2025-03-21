#!/usr/bin/env python3
import argparse
import os
import time
from typing import Optional
import pygame
from tqdm import tqdm

from environment.board import Board, Direction
from agent.q_agent import QLearningAgent
from visualization.display import Display
from utils.performance_tracker import PerformanceTracker

def parse_args():
    parser = argparse.ArgumentParser(description='Learn2Slither - A snake that learns')
    parser.add_argument('-sessions', type=int, default=100,
                       help='Number of training sessions')
    parser.add_argument('-save', type=str,
                       help='Path to save the trained model')
    parser.add_argument('-load', type=str,
                       help='Path to load a trained model')
    parser.add_argument('-visual', type=str, choices=['on', 'off'], default='on',
                       help='Enable or disable visualization')
    parser.add_argument('-dontlearn', action='store_true',
                       help='Disable learning (use with -load)')
    parser.add_argument('-step-by-step', action='store_true',
                       help='Enable step-by-step mode')
    parser.add_argument('-speed', type=float, default=0.1,
                       help='Game speed in seconds (lower is faster)')
    parser.add_argument('-curriculum', action='store_true',
                       help='Enable curriculum learning')
    return parser.parse_args()

def create_model_dir():
    """Create models directory if it doesn't exist."""
    if not os.path.exists('models'):
        os.makedirs('models')

def create_agent(board_size):
    state_size = board_size * 4  # 4 directions * board_size cells per direction
    action_size = 4  # Up, Down, Left, Right
    return QLearningAgent(state_size, action_size)

def state_to_tensor(state, board_size):
    """Convert state representation to tensor."""
    state_vector = []
    for direction in state:
        # Pad or truncate each direction's view to board_size
        view = direction[:board_size]
        while len(view) < board_size:
            view.append('W')
            
        for cell in view:
            if cell == '0':
                state_vector.append(0.0)
            elif cell == 'H':
                state_vector.append(1.0)
            elif cell == 'S':
                state_vector.append(0.5)
            elif cell == 'G':
                state_vector.append(0.8)
            elif cell == 'R':
                state_vector.append(0.3)
            elif cell == 'W':
                state_vector.append(-1.0)
    
    return state_vector

def run_session(board, agent, display=None, speed=0.1, tracker=None):
    """Run a single game session."""
    state = board._get_state()
    max_length = 1
    steps = 0
    
    while True:
        if display:
            # Utiliser la méthode draw_board qui gère déjà l'affichage Pygame
            quit_game = display.draw_board(board.board, len(board.snake_positions), max_length)
            if quit_game:
                return max_length, steps
            time.sleep(speed)
        
        # Convert state to tensor format
        state_tensor = state_to_tensor(state, board.size)
        action = agent.get_action(state_tensor)
        direction = Direction(action)
        
        # Take action and observe next state
        next_state, reward, done, length = board.step(direction)
        
        if not args.dontlearn:
            # Convert next_state to tensor format for training
            next_state_tensor = state_to_tensor(next_state, board.size)
            agent.train(state_tensor, action, reward, next_state_tensor, done)
        
        state = next_state
        steps += 1
        max_length = max(max_length, length)
        
        if tracker:
            tracker.update(reward, length, steps)
        
        if done:
            break
    
    return max_length, steps

def run_curriculum(args, agent: QLearningAgent, display: Optional[Display]):
    """Run curriculum learning with progressively increasing board sizes."""
    tracker = PerformanceTracker()
    stages = [
        {'size': 5, 'sessions': args.sessions // 4},
        {'size': 7, 'sessions': args.sessions // 4},
        {'size': 10, 'sessions': args.sessions // 2}
    ]
    
    for stage in stages:
        print(f"\nStarting stage with board size {stage['size']}")
        board = Board(size=stage['size'])
        
        for session in tqdm(range(stage['sessions']), desc=f"Training (size={stage['size']})"):
            max_length, steps = run_session(board, agent, display, args.speed, tracker)
            if session % 10 == 0:
                stats = tracker.get_stats()
                print(f"\nBoard size {stage['size']}, Session {session}")
                print(f"Max length = {max_length}, Duration = {steps}")
                print(f"Average length = {stats['avg_length']:.2f}")
                print(f"Average reward = {stats['avg_reward']:.2f}")
                print(f"Average apples = {stats['avg_apples']:.2f}")

def main():
    # Parse command line arguments
    global args
    args = parse_args()
    
    # Create model directory if it doesn't exist
    if args.save:
        create_model_dir()
    
    # Initialize board and agent
    board = Board(size=10)
    agent = create_agent(board.size)
    
    # Load pre-trained model if specified
    if args.load:
        print(f"Load trained model from {args.load}")
        agent.load_model(args.load)
    
    # Initialize display if visual mode is on
    display = None
    if args.visual == "on":
        display = Display(board.size)
    
    # Initialize performance tracker
    tracker = PerformanceTracker() if not args.dontlearn else None
    
    # Training loop
    progress_bar = tqdm(range(args.sessions), desc="Training")
    for _ in progress_bar:
        board.reset()
        max_length, steps = run_session(board, agent, display, args.speed, tracker)
        
        if tracker:
            progress_bar.set_postfix({
                'max_length': max_length,
                'steps': steps,
                'avg_reward': tracker.get_average_reward(),
                'avg_length': tracker.get_average_length()
            })
    
    # Save trained model if specified
    if args.save:
        agent.save_model(args.save)
        print(f"Saved trained model to {args.save}")

if __name__ == "__main__":
    main() 