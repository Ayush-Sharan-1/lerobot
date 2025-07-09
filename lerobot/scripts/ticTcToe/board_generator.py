#!/usr/bin/env python3
"""
Tic-Tac-Toe Board Generator for Robot Training Dataset
Generates diverse board configurations for each target position.
"""

import random
import itertools
from typing import List, Tuple, Optional

class TicTacToeGenerator:
    def __init__(self, seed: int = 0):
        # Grid positions (0-8, left to right, top to bottom)
        # 0 1 2
        # 3 4 5  
        # 6 7 8
        self.positions = list(range(9))
        random.seed(seed)
        
    def create_board_config(self, empty_position: int, num_x: int, num_o: int) -> List[Optional[str]]:
        """
        Create a board configuration with specified empty position and piece counts.
        
        Args:
            empty_position: Position that must remain empty (0-8)
            num_x: Number of X pieces to place
            num_o: Number of O pieces to place
            
        Returns:
            List representing the board state (None for empty, 'X' for cross, 'O' for circle)
        """
        board = [None] * 9
        available_positions = [i for i in self.positions if i != empty_position]
        
        # Randomly select positions for X and O
        selected_positions = random.sample(available_positions, num_x + num_o)
        
        # Place X pieces first, then O pieces
        for i in range(num_x):
            board[selected_positions[i]] = 'X'
        for i in range(num_x, num_x + num_o):
            board[selected_positions[i]] = 'O'
            
        return board
    
    def is_valid_game_state(self, board: List[Optional[str]]) -> bool:
        """
        Check if the board represents a valid tic-tac-toe game state.
        X goes first, so X count should be equal to O count or O count + 1.
        Also checks that no winning condition exists.
        """
        x_count = board.count('X')
        o_count = board.count('O')
        
        # Check piece count validity (X goes first)
        if not (x_count == o_count or x_count == o_count + 1):
            return False
            
        # Check for winning conditions
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        
        for combo in winning_combinations:
            if (board[combo[0]] == board[combo[1]] == board[combo[2]] and 
                board[combo[0]] is not None):
                return False
        
        return True
    
    def generate_configurations(self, empty_position: int, num_configs: int = 10) -> List[List[Optional[str]]]:
        """
        Generate diverse valid board configurations for a given empty position.
        
        Args:
            empty_position: Position that must remain empty (0-8)
            num_configs: Number of configurations to generate
            
        Returns:
            List of board configurations
        """
        configurations = []
        max_attempts = 1000  # Prevent infinite loops
        
        # Define different scenarios to ensure diversity
        base_scenarios = [
            (0, 0),  # Empty board
            (1, 0),  # Just one X
            (1, 1),  # One X, one O
            (2, 1),  # Two X, one O
            (2, 2),  # Two X, two O
            (3, 2),  # Three X, two O
            (3, 3),  # Three X, three O
            (4, 3),  # Four X, three O
            (4, 4),  # Four X, four O
        ]
        
        # TODO: Need to fix minor bug. If attempts > max_attempts that scenario will be skipped.
        # Outer loop: continue until we have enough configurations
        while len(configurations) < num_configs:
            available_scenarios = base_scenarios.copy()
            
            # Inner loop: randomly pick and pop scenarios
            while available_scenarios and len(configurations) < num_configs:
                # Randomly pick and remove a scenario
                scenario_index = random.randint(0, len(available_scenarios) - 1)
                num_x, num_o = available_scenarios.pop(scenario_index)
                
                # Try to generate a valid configuration for this scenario
                attempts = 0
                while attempts < max_attempts:
                    board = self.create_board_config(empty_position, num_x, num_o)
                    if self.is_valid_game_state(board):
                        configurations.append(board)
                        break
                    attempts += 1
        
        return configurations
    
    def print_board(self, board: List[Optional[str]]) -> None:
        """Print board in a readable format."""
        symbols = {'X': 'B', 'O': 'W', None: '.'}
        for i in range(3):
            row = ' '.join(symbols[board[i*3 + j]] for j in range(3))
            print(row)
        print()
    
    def board_to_description(self, board: List[Optional[str]]) -> str:
        """Convert board to human-readable description."""
        x_count = board.count('X')
        o_count = board.count('O')
        
        if x_count == 0 and o_count == 0:
            return "Empty board"
        
        return f"{x_count} B's, {o_count} W's"

# Example usage
if __name__ == "__main__":

    seed = 1
    generator = TicTacToeGenerator(seed = seed)

    # Grid positions (0-8, left to right, top to bottom)
    # 1 2 3
    # 4 5 6  
    # 5 8 9

    empty_position = 9
    num_configs=10
    
    # Generate 10 configurations for top-right corner (position 2)
    print(f"=== {num_configs} Board Configurations for Position {empty_position} ===\n")
    
    configs = generator.generate_configurations(empty_position=empty_position-1, num_configs=num_configs)
    
    for i, board in enumerate(configs, 1):
        print(f"Configuration {i}: {generator.board_to_description(board)}")
        generator.print_board(board)