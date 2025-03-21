# Learn2Slither

A reinforcement learning project where a snake learns to navigate and survive in a grid environment through Q-learning.

## Features

- 10x10 grid environment
- Snake with Q-learning capabilities
- Multiple training modes
- Visual interface with step-by-step mode
- Model saving and loading
- Configurable training sessions

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the game with different options:

```bash
# Train for 10 sessions and save the model
./snake -sessions 10 -save models/10sess.txt -visual off

# Load a trained model and run in visual mode
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn

# Run in step-by-step mode
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```

## Project Structure

- `snake.py`: Main entry point
- `environment/`: Contains the game board and rules
- `agent/`: Q-learning agent implementation
- `visualization/`: Pygame-based visualization
- `models/`: Directory for saved model states 