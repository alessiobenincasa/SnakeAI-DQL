import numpy as np
from enum import Enum
from typing import Tuple, List, Optional

class CellType(Enum):
    EMPTY = 0
    SNAKE_HEAD = 1
    SNAKE_BODY = 2
    GREEN_APPLE = 3
    RED_APPLE = 4

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Board:
    def __init__(self, size: int = 10):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.snake_positions: List[Tuple[int, int]] = []
        self.green_apples: List[Tuple[int, int]] = []
        self.red_apple: Optional[Tuple[int, int]] = None
        self.current_direction = Direction.RIGHT
        self.prev_distance_to_green = float('inf')
        self.reset()

    def reset(self) -> None:
        """Reset the board to initial state."""
        self.board.fill(CellType.EMPTY.value)
        self._init_snake()
        self._place_apples()
        self.prev_distance_to_green = self._get_min_distance_to_green()
        return self._get_state()

    def _init_snake(self) -> None:
        """Initialize snake with length 3 at random position."""
        self.snake_positions = []
        # Random starting position for head
        head_x = np.random.randint(2, self.size - 2)
        head_y = np.random.randint(2, self.size - 2)
        
        # Initialize snake with 3 segments
        self.snake_positions = [
            (head_x, head_y),
            (head_x - 1, head_y),
            (head_x - 2, head_y)
        ]
        
        # Place snake on board
        self.board[head_x, head_y] = CellType.SNAKE_HEAD.value
        for x, y in self.snake_positions[1:]:
            self.board[x, y] = CellType.SNAKE_BODY.value

    def _place_apples(self) -> None:
        """Place 2 green apples and 1 red apple randomly."""
        empty_cells = list(zip(*np.where(self.board == CellType.EMPTY.value)))
        
        if len(empty_cells) < 3:
            return  # Not enough space for apples
            
        # Place green apples
        self.green_apples = []
        for _ in range(2):
            if empty_cells:
                idx = np.random.randint(len(empty_cells))
                pos = empty_cells.pop(idx)
                self.green_apples.append(pos)
                self.board[pos] = CellType.GREEN_APPLE.value
        
        # Place red apple
        if empty_cells:
            idx = np.random.randint(len(empty_cells))
            self.red_apple = empty_cells[idx]
            self.board[self.red_apple] = CellType.RED_APPLE.value

    def _get_state(self) -> List[List[str]]:
        """Get the snake's vision in all 4 directions."""
        # If snake has no length, return empty state
        if not self.snake_positions:
            return [['W'] for _ in range(4)]
            
        head_x, head_y = self.snake_positions[0]
        vision = []
        
        # Check all 4 directions
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        
        for dx, dy in directions:
            view = []
            x, y = head_x, head_y
            while 0 <= x < self.size and 0 <= y < self.size:
                cell = self.board[x, y]
                if cell == CellType.EMPTY.value:
                    view.append('0')
                elif cell == CellType.SNAKE_HEAD.value:
                    view.append('H')
                elif cell == CellType.SNAKE_BODY.value:
                    view.append('S')
                elif cell == CellType.GREEN_APPLE.value:
                    view.append('G')
                elif cell == CellType.RED_APPLE.value:
                    view.append('R')
                x, y = x + dx, y + dy
            
            if len(view) < self.size:
                view.append('W')  # Wall
            vision.append(view)
            
        return vision

    def _get_min_distance_to_green(self) -> float:
        """Calculate minimum Manhattan distance to any green apple."""
        if not self.green_apples or not self.snake_positions:
            return float('inf')
            
        head_x, head_y = self.snake_positions[0]
        min_distance = float('inf')
        
        for apple_x, apple_y in self.green_apples:
            distance = abs(head_x - apple_x) + abs(head_y - apple_y)
            min_distance = min(min_distance, distance)
            
        return min_distance

    def step(self, action: Direction) -> Tuple[List[List[str]], float, bool, int]:
        """
        Execute one step in the environment.
        
        Args:
            action: The direction to move the snake
            
        Returns:
            state: The new state after the action
            reward: The reward for the action
            done: Whether the episode is finished
            length: Current length of the snake
        """
        # If snake has no length, end the game
        if not self.snake_positions:
            return self._get_state(), -10.0, True, 0
            
        # Get head position
        head_x, head_y = self.snake_positions[0]
        
        # Calculate new head position
        if action == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif action == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        elif action == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        else:  # LEFT
            new_head = (head_x - 1, head_y)
            
        # Check if game over
        if self._is_collision(new_head):
            return self._get_state(), -10.0, True, len(self.snake_positions)
            
        # Calculate distance to nearest green apple
        current_distance = self._get_min_distance_to_green()
        distance_reward = 0.0
        
        if current_distance < self.prev_distance_to_green:
            distance_reward = 0.1  # Small reward for moving closer to green apple
        
        self.prev_distance_to_green = current_distance
            
        # Initialize reward
        reward = 0.0
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        # Add distance reward
        reward += distance_reward
        
        grow = False
        shrink = False
        
        # Check for apples
        if new_head in self.green_apples:
            # Bigger reward for longer snake
            base_reward = 1.0
            length_multiplier = len(self.snake_positions) / 3.0  # Normalize by starting length
            reward += base_reward * length_multiplier
            grow = True
            self.green_apples.remove(new_head)
        elif new_head == self.red_apple:
            reward -= 1.0  # Fixed penalty for red apple
            shrink = True
            self.red_apple = None
            
        # Update snake position
        self.snake_positions.insert(0, new_head)
        if shrink:
            if len(self.snake_positions) >= 2:
                self.snake_positions.pop()
                self.snake_positions.pop()  # Remove an extra segment
            else:
                self.snake_positions = []  # Snake length becomes 0
        elif not grow:
            self.snake_positions.pop()
            
        # Check if snake length is 0
        if len(self.snake_positions) == 0:
            return self._get_state(), -10.0, True, 0
            
        # Survival reward (small positive reward for staying alive)
        reward += 0.01
        
        # Update board
        self.board.fill(CellType.EMPTY.value)
        
        # Place snake
        if self.snake_positions:
            self.board[new_head] = CellType.SNAKE_HEAD.value
            for pos in self.snake_positions[1:]:
                self.board[pos] = CellType.SNAKE_BODY.value
            
        # Make sure apples are always visible
        for apple_pos in self.green_apples:
            self.board[apple_pos] = CellType.GREEN_APPLE.value
        if self.red_apple:
            self.board[self.red_apple] = CellType.RED_APPLE.value
        
        # Replace eaten apples
        if grow or shrink:
            self._place_apples()
            
        return self._get_state(), reward, False, len(self.snake_positions)

    def _is_collision(self, position: Tuple[int, int]) -> bool:
        """Check if the position collides with wall or snake body."""
        x, y = position
        # Check wall collision
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        # Check self collision (excluding tail which will move)
        if position in self.snake_positions[:-1]:
            return True
        return False 