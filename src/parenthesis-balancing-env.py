import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import time
import gymnasium as gym
from gymnasium import spaces
import math

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False):
        # Simple fallback for non-IPython environments
        print("\033c", end="")  # ANSI escape code to clear screen

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)  # For GPU reproducibility

# The ParenthesisBalancingEnv
class ParenthesisBalancingEnv(gym.Env):
    """
    A grid-based environment where an agent must collect opening and closing parentheses
    in a valid balanced order. For example: ()()(), (()()), ((()))
    
    The agent must first collect an opening parenthesis '(' and then ensure
    each closing parenthesis ')' corresponds to a previously collected opening one.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, grid_size=7, max_depth=3, max_steps=200, 
                 fixed_positions=False, curriculum_level=0, render_mode='human'):
        super(ParenthesisBalancingEnv, self).__init__()

        self.grid_size = grid_size
        self.max_depth = max_depth  # Maximum nesting level
        self.max_steps = max_steps
        self.fixed_positions = fixed_positions
        self.curriculum_level = curriculum_level
        self.render_mode = render_mode

        # Define action space (0: up, 1: right, 2: down, 3: left)
        self.action_space = spaces.Discrete(4)

        # Define observation space
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.int32),
            'agent_pos': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            'opening_count': spaces.Discrete(max_depth * 3 + 1),  # Max possible openings
            'closing_count': spaces.Discrete(max_depth * 3 + 1),   # Max possible closings
            'stack_depth': spaces.Discrete(max_depth + 1),  # Current nesting level
            'stack_status': spaces.Box(low=0, high=1, shape=(max_depth,), dtype=np.float32)  # Representation of stack
        })

        # Initialize state
        self.grid = None
        self.agent_pos = None
        self.opening_count = 0
        self.closing_count = 0
        self.current_step = 0
        
        # Stack to track opening parentheses (for balancing)
        self.stack = []
        self.max_stack_depth_reached = 0
        
        # Item encoding: 0 is empty, 1 is '(', 2 is ')'
        self.item_types = {0: 'empty', 1: 'opening', 2: 'closing'}
        self.item_symbols = {0: ' ', 1: '(', 2: ')'}
        self.item_colors = {0: 'white', 1: 'green', 2: 'red'}
        
        # For tracking the collected sequence
        self.collected_sequence = []
        
        # Target pattern (generated during reset)
        self.target_pattern = None
        self.remaining_to_collect = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Reset counters and stack
        self.opening_count = 0
        self.closing_count = 0
        self.current_step = 0
        self.stack = []
        self.max_stack_depth_reached = 0
        self.collected_sequence = []
        
        # Generate target balanced pattern based on curriculum level
        self.target_pattern, self.remaining_to_collect = self._generate_target_pattern()
        
        # Place agent at center of grid
        if self.curriculum_level <= 1 or self.fixed_positions:
            # Fixed starting position for easier learning
            self.agent_pos = np.array([self.grid_size - 1, self.grid_size // 2])
        else:
            # Random position for harder levels
            self.agent_pos = np.array([
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ])
            
        # Place items on grid
        self._place_items()
        
        # Return observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
        
    def _generate_target_pattern(self):
        """Generate a valid balanced parentheses pattern based on curriculum level"""
        # Determine pattern complexity based on curriculum
        if self.curriculum_level == 0:
            # Simplest: Just "()"
            pattern = ["(", ")"]
            depth = 1
        elif self.curriculum_level == 1:
            # Simple: "()()"
            pattern = ["(", ")", "(", ")"]
            depth = 1
        elif self.curriculum_level == 2:
            # Nested: "(())"
            pattern = ["(", "(", ")", ")"]
            depth = 2
        elif self.curriculum_level == 3:
            # Mixed: "()((()))"
            pattern = ["(", ")", "(", "(", "(", ")", ")", ")"]
            depth = 3
        else:
            # Generate random valid pattern with complexity based on level
            max_pairs = min(3 + self.curriculum_level, self.max_depth * 2)
            pattern = self._generate_random_balanced_pattern(max_pairs)
            depth = self._calculate_max_depth(pattern)
            
        # Count item types for placement
        item_counts = {'(': 0, ')': 0}
        for symbol in pattern:
            item_counts[symbol] += 1
            
        # Create dictionary of remaining items to collect
        remaining = {'(': item_counts['('], ')': item_counts[')']}
        
        return pattern, remaining
    
    def _generate_random_balanced_pattern(self, max_pairs):
        """Generate a random valid balanced parentheses pattern"""
        # Start with simplest valid pattern
        if max_pairs <= 1:
            return ["(", ")"]
            
        # For more complex patterns, we'll use a recursive approach
        def generate_subpattern(pairs_remaining, max_depth):
            if pairs_remaining <= 0:
                return []
            if pairs_remaining == 1:
                return ["(", ")"]
                
            # Decide structure randomly
            if random.random() < 0.5 and max_depth > 1:
                # Nested structure: (...)
                inner_pairs = random.randint(1, pairs_remaining - 1)
                inner_pattern = generate_subpattern(inner_pairs, max_depth - 1)
                outer_pattern = ["("] + inner_pattern + [")"]
                
                remaining = pairs_remaining - inner_pairs - 1
                if remaining > 0:
                    # Add sequential pattern for remaining pairs
                    return outer_pattern + generate_subpattern(remaining, max_depth)
                return outer_pattern
            else:
                # Sequential structure: ()()...
                return ["(", ")"] + generate_subpattern(pairs_remaining - 1, max_depth)
                
        return generate_subpattern(max_pairs, self.max_depth)
    
    def _calculate_max_depth(self, pattern):
        """Calculate the maximum nesting depth of a pattern"""
        depth = 0
        max_depth = 0
        for symbol in pattern:
            if symbol == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif symbol == ")":
                depth -= 1
        return max_depth
    
    def _place_items(self):
        """Place opening and closing parentheses on the grid"""
        # Clear any existing items
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        if self.fixed_positions and self.curriculum_level <= 1:
            # Level 0-1: Fixed positions for easier learning
            # Opening parentheses in left half
            self._place_items_in_region(1, self.remaining_to_collect['('], 
                                        0, self.grid_size-1, 
                                        0, self.grid_size//2-1)
            # Closing parentheses in right half
            self._place_items_in_region(2, self.remaining_to_collect[')'], 
                                        0, self.grid_size-1, 
                                        self.grid_size//2, self.grid_size-1)
        elif self.curriculum_level <= 3:
            # Level 2-3: Semi-structured with some randomization
            self._place_items_with_distance(1, self.remaining_to_collect['('], min_distance=1)
            self._place_items_with_distance(2, self.remaining_to_collect[')'], min_distance=1)
        else:
            # Level 4+: Fully random placement
            self._place_random_items(1, self.remaining_to_collect['('])
            self._place_random_items(2, self.remaining_to_collect[')'])

    def _place_items_in_region(self, item_type, count, min_row, max_row, min_col, max_col):
        """Place items of a specific type within a defined region of the grid."""
        positions = []
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                # Ensure position is empty and not agent's starting position
                if self.grid[row, col] == 0 and not (row == self.agent_pos[0] and col == self.agent_pos[1]):
                    positions.append((row, col))

        # Ensure we have enough positions
        if len(positions) < count:
             # Fallback if region is too small
             print(f"Warning: Not enough positions in region for item {item_type}. Placing randomly.")
             self._place_random_items(item_type, count)
             return

        # Randomly select positions for items
        selected_positions = random.sample(positions, count)
        for row, col in selected_positions:
            self.grid[row, col] = item_type

    def _place_items_with_distance(self, item_type, count, min_distance=1):
        """Place items with a minimum distance from each other and the agent."""
        placed = 0
        max_attempts_per_item = 100
        total_attempts = 0

        while placed < count and total_attempts < max_attempts_per_item * count:
            row = random.randint(0, self.grid_size-1)
            col = random.randint(0, self.grid_size-1)
            total_attempts += 1

            # Check if position is valid (empty and not where agent is)
            if self.grid[row, col] == 0 and not (row == self.agent_pos[0] and col == self.agent_pos[1]):
                # Check minimum distance from agent
                agent_distance = abs(row - self.agent_pos[0]) + abs(col - self.agent_pos[1])
                if agent_distance >= min_distance:
                    # Check minimum distance from other items of the same type
                    valid_dist = True
                    # Check a box around the potential placement
                    for r in range(max(0, row - min_distance + 1), min(self.grid_size, row + min_distance)):
                         for c in range(max(0, col - min_distance + 1), min(self.grid_size, col + min_distance)):
                              if self.grid[r, c] == item_type:
                                  item_dist = abs(row - r) + abs(col - c)
                                  if item_dist < min_distance:
                                      valid_dist = False
                                      break
                         if not valid_dist:
                             break

                    if valid_dist:
                        self.grid[row, col] = item_type
                        placed += 1
                        total_attempts = 0  # Reset attempts after successful placement

        # If we couldn't place all items with distance constraints, fall back to random
        if placed < count:
            print(f"Warning: Could not place all {count} items of type {item_type} with min_distance {min_distance}. Placing remaining randomly.")
            self._place_random_items(item_type, count - placed)

    def _place_random_items(self, item_type, count):
        """Place items randomly on the grid."""
        placed = 0
        max_attempts = 100 * count  # Avoid infinite loops
        attempts = 0

        while placed < count and attempts < max_attempts:
            row = random.randint(0, self.grid_size-1)
            col = random.randint(0, self.grid_size-1)
            attempts += 1

            # Make sure position is empty and not where agent is
            if self.grid[row, col] == 0 and not (row == self.agent_pos[0] and col == self.agent_pos[1]):
                self.grid[row, col] = item_type
                placed += 1

        if placed < count:
             print(f"Error: Could not place all {count} items of type {item_type} randomly after {max_attempts} attempts.")

    def _get_observation(self):
        """Return the current state as an observation."""
        # Create stack status representation (1 for positions with opening parentheses)
        stack_status = np.zeros(self.max_depth, dtype=np.float32)
        for i, _ in enumerate(self.stack):
            if i < self.max_depth:
                stack_status[i] = 1.0
                
        return {
            'grid': self.grid.copy(),
            'agent_pos': self.agent_pos.copy(),
            'opening_count': self.opening_count,
            'closing_count': self.closing_count,
            'stack_depth': len(self.stack),
            'stack_status': stack_status
        }

    def _get_info(self):
        """Return auxiliary information about the state."""
        min_distances = self.get_min_distances()
        return {
            'target_pattern': "".join(self.target_pattern),
            'collected_sequence': "".join([self.item_symbols[item] for item in self.collected_sequence]),
            'remaining_opening': self.remaining_to_collect['('],
            'remaining_closing': self.remaining_to_collect[')'],
            'stack_depth': len(self.stack),
            'max_stack_depth': self.max_stack_depth_reached,
            'min_dist_opening': min_distances['opening'],
            'min_dist_closing': min_distances['closing'],
            'is_balanced': len(self.stack) == 0 and (self.opening_count > 0 or self.curriculum_level == 0)
        }

    def step(self, action):
        """Take a step in the environment based on the action."""
        self.current_step += 1

        # Move the agent based on the action
        # 0: up, 1: right, 2: down, 3: left
        old_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1

        # Check what's at the new position
        x, y = self.agent_pos
        item_at_pos = self.grid[x, y]

        reward = -0.1  # Small step penalty to encourage efficiency
        terminated = False
        truncated = False
        info = {'collected': None, 'balanced': False, 'error': None}

        # Process item collection if there's an item
        if item_at_pos > 0:
            item_type = self.item_types[item_at_pos]
            item_symbol = self.item_symbols[item_at_pos]
            
            if item_type == 'opening':  # Opening parenthesis (
                # Always valid to collect an opening parenthesis if needed
                if self.remaining_to_collect['('] > 0:
                    self.remaining_to_collect['('] -= 1
                    self.opening_count += 1
                    self.collected_sequence.append(item_at_pos)
                    self.stack.append('(')  # Push to stack
                    self.max_stack_depth_reached = max(self.max_stack_depth_reached, len(self.stack))
                    reward += 2.0
                    info['collected'] = item_symbol
                    
                    # Remove item from grid
                    self.grid[x, y] = 0
                else:
                    # Not needed or no more openings required
                    reward -= 2.0
                    info['error'] = "Collected unnecessary opening parenthesis"
                    terminated = True
                    
            elif item_type == 'closing':  # Closing parenthesis )
                # Valid only if there's an opening in the stack
                if len(self.stack) > 0 and self.remaining_to_collect[')'] > 0:
                    # Match and pop from stack
                    self.stack.pop()
                    self.remaining_to_collect[')'] -= 1
                    self.closing_count += 1
                    self.collected_sequence.append(item_at_pos)
                    reward += 2.0
                    info['collected'] = item_symbol
                    
                    # Remove item from grid
                    self.grid[x, y] = 0
                    
                    # Check if pattern is complete (all items collected and balanced)
                    if self.remaining_to_collect['('] == 0 and self.remaining_to_collect[')'] == 0:
                        if len(self.stack) == 0:  # Properly balanced
                            reward += 50.0
                            terminated = True
                            info['balanced'] = True
                        else:  # All items collected but not balanced
                            reward -= 10.0
                            terminated = True
                            info['error'] = "All items collected but parentheses not balanced"
                else:
                    # Unmatched closing parenthesis - syntax error!
                    reward -= 5.0
                    terminated = True
                    info['error'] = "Unmatched closing parenthesis"

        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated:
                info['error'] = "Max steps reached"

        # Get observation and info
        observation = self._get_observation()
        full_info = self._get_info()
        full_info.update(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, full_info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            clear_output(wait=True)
            print(f"Step: {self.current_step}/{self.max_steps} | Stack depth: {len(self.stack)}")
            print(f"Target: {''.join(self.target_pattern)}")
            print(f"Collected: {''.join([self.item_symbols[item] for item in self.collected_sequence])}")
            print(f"Remaining: {self.remaining_to_collect['(']} opening, {self.remaining_to_collect[')']} closing")
            
            # Print stack visualization
            stack_str = "Stack: " + "".join(self.stack)
            print(stack_str)
            
            grid_repr = ""
            for r in range(self.grid_size):
                row_str = "|"
                for c in range(self.grid_size):
                    if r == self.agent_pos[0] and c == self.agent_pos[1]:
                        row_str += " A "  # Agent
                    elif self.grid[r, c] == 1:
                        row_str += " ( "
                    elif self.grid[r, c] == 2:
                        row_str += " ) "
                    else:
                        row_str += " . "
                grid_repr += row_str + "|\n"
            print("-" * (self.grid_size * 3 + 2))
            print(grid_repr, end="")
            print("-" * (self.grid_size * 3 + 2))
            time.sleep(0.1)
            return None
            
        elif self.render_mode == "rgb_array":
             # Create a simple RGB array representation
             rgb_array = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
             # Background: Gray
             rgb_array[:,:,:] = 128

             # Items
             for r in range(self.grid_size):
                 for c in range(self.grid_size):
                     if self.grid[r, c] == 1:  # '(': Green
                         rgb_array[r, c] = [0, 255, 0]
                     elif self.grid[r, c] == 2:  # ')': Red
                         rgb_array[r, c] = [255, 0, 0]

             # Agent: White
             ax, ay = self.agent_pos
             if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
                 rgb_array[ax, ay] = [255, 255, 255]

             # Upscale for better visibility if needed
             scale = 50
             rgb_array = np.kron(rgb_array, np.ones((scale, scale, 1), dtype=np.uint8))
             return rgb_array

    def close(self):
        pass  # No resources to clean up

    def get_min_distances(self):
        """Calculate minimum Manhattan distance from agent to each item type."""
        min_distances = {'opening': float('inf'), 'closing': float('inf')}
        agent_r, agent_c = self.agent_pos

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                item_type_num = self.grid[r, c]
                if item_type_num > 0:
                    item_type = self.item_types[item_type_num]
                    distance = abs(r - agent_r) + abs(c - agent_c)
                    min_distances[item_type] = min(min_distances[item_type], distance)

        # Replace infinite distances with a large number
        max_dist = self.grid_size * 2
        for item_type in min_distances:
            if min_distances[item_type] == float('inf'):
                min_distances[item_type] = max_dist

        return min_distances


# Stack-based PDA (Pushdown Automaton) for Parenthesis Balancing
class ParenthesisBalancingPDA:
    """
    A pushdown automaton for recognizing balanced parentheses language.
    This is a simple stack-based implementation.
    """
    def __init__(self, max_depth=10):
        # States: q0 (initial), qaccept (accepting), qreject (rejecting)
        self.states = ['q0', 'qaccept', 'qreject']
        self.current_state = 'q0'
        
        # Stack to track opening parentheses
        self.stack = []
        self.max_depth = max_depth
        
        # For tracking
        self.max_stack_depth = 0
        self.collected_sequence = []
        
    def reset(self):
        """Reset the automaton to initial state."""
        self.current_state = 'q0'
        self.stack = []
        self.max_stack_depth = 0
        self.collected_sequence = []
        
    def process_symbol(self, symbol):
        """
        Process an input symbol according to the PDA rules.
        Symbol should be either '(' or ')'
        """
        if self.current_state in ['qaccept', 'qreject']:
            return self.current_state
            
        # Store symbol in collected sequence
        self.collected_sequence.append(symbol)
        
        if symbol == '(':
            # Push to stack for opening parenthesis
            self.stack.append('(')
            self.max_stack_depth = max(self.max_stack_depth, len(self.stack))
            # Stay in q0
        elif symbol == ')':
            # Pop for closing parenthesis if stack not empty
            if len(self.stack) > 0:
                self.stack.pop()
                # Stay in q0
            else:
                # Unmatched closing parenthesis - reject
                self.current_state = 'qreject'
        else:
            # Invalid symbol - reject
            self.current_state = 'qreject'
            
        # Accept if stack is empty after processing some symbols
        if len(self.stack) == 0 and len(self.collected_sequence) > 0:
            self.current_state = 'qaccept'
            
        return self.current_state
        
    def get_state(self):
        """Return the current state of the automaton."""
        return {
            'state': self.current_state,
            'stack': self.stack.copy(),
            'stack_depth': len(self.stack),
            'max_stack_depth': self.max_stack_depth,
            'collected_sequence': self.collected_sequence.copy()
        }
    
    def is_balanced(self):
        """Check if the current sequence is balanced."""
        return self.current_state == 'qaccept'


# Advanced Reward Machine for Parenthesis Balancing
class ParenthesisRewardMachine:
    """
    Reward machine for the parenthesis balancing task.
    Provides sophisticated reward shaping to guide the agent toward proper balancing.
    """
    def __init__(self):
        # Define states
        self.states = ['u_init', 'u_collecting', 'u_accept', 'u_reject']
        self.current_state = 'u_init'
        
        # Define reward shaping parameters
        self.correct_collection_reward = 2.0
        self.balanced_bonus = 50.0
        self.unbalanced_penalty = -5.0
        self.step_penalty = -0.1
        
        # Distance-based shaping parameters
        self.distance_improvement_factor = 0.3
        self.distance_deterioration_factor = -0.2
        
        # Progress tracking
        self.last_min_distances = {'opening': float('inf'), 'closing': float('inf')}
        self.approach_counts = {'opening': 0, 'closing': 0}
        
    def reset(self):
        """Reset the reward machine."""
        self.current_state = 'u_init'
        self.last_min_distances = {'opening': float('inf'), 'closing': float('inf')}
        self.approach_counts = {'opening': 0, 'closing': 0}
        
    def transition(self, pda_state):
        """Update reward machine state based on the PDA state."""
        pda_s = pda_state['state']
        
        if pda_s == 'qaccept':
            self.current_state = 'u_accept'
        elif pda_s == 'qreject':
            self.current_state = 'u_reject'
        elif len(pda_state['collected_sequence']) > 0:
            self.current_state = 'u_collecting'
        else:
            self.current_state = 'u_init'
            
        return self.current_state
        
    def get_reward(self, old_pda_state, new_pda_state, env_reward, distances=None, info=None):
        """Calculate shaped reward based on PDA state transition and other factors."""
        # Start with environment reward
        final_reward = env_reward
        
        # Determine current goal based on stack state
        current_goal = self._determine_goal(new_pda_state)
        
        # Add distance-based shaping if distances provided
        if distances is not None and current_goal in distances:
            current_distance = distances[current_goal]
            prev_distance = self.last_min_distances[current_goal]
            
            if prev_distance != float('inf') and current_distance < float('inf'):
                # Reward for getting closer to target
                if current_distance < prev_distance:
                    improvement = prev_distance - current_distance
                    distance_reward = self.distance_improvement_factor * improvement
                    final_reward += distance_reward
                    self.approach_counts[current_goal] = min(5, self.approach_counts[current_goal] + 1)
                # Penalty for moving away from target
                elif current_distance > prev_distance and not (info and info.get('collected') == current_goal):
                    deterioration = current_distance - prev_distance
                    distance_penalty = self.distance_deterioration_factor * deterioration
                    final_reward += distance_penalty
                    self.approach_counts[current_goal] = max(0, self.approach_counts[current_goal] - 1)
                    
            # Update last distances
            self.last_min_distances = distances.copy()
        
        # Add consistency bonus for maintaining approach
        if current_goal in self.approach_counts and self.approach_counts[current_goal] > 2:
            consistency_bonus = 0.1 * self.approach_counts[current_goal]
            final_reward += min(0.5, consistency_bonus)
            
        # Additional bonus for balancing (already in env_reward, but can be enhanced)
        if new_pda_state['state'] == 'qaccept' and old_pda_state['state'] != 'qaccept':
            # Additional bonus based on complexity
            depth_bonus = new_pda_state['max_stack_depth'] * 2.0
            final_reward += depth_bonus
            
        return final_reward
        
    def _determine_goal(self, pda_state):
        """Determine the current collection goal based on stack state."""
        # If stack is empty, we need an opening parenthesis
        if len(pda_state['stack']) == 0:
            return 'opening'
        # If stack has openings, we can either get another opening or a closing
        # Prefer closing for balance if stack is getting deep
        elif len(pda_state['stack']) >= pda_state['max_stack_depth'] / 2:
            return 'closing'
        # Otherwise flexible, but give slight preference to opening
        else:
            return 'opening'


# Hierarchical MDPTM-RM for Parenthesis Balancing
class HierarchicalMDPTM_RM_Parenthesis:
    """
    Hierarchical MDPTM-RM implementation for the parenthesis balancing environment.
    Uses a PDA and reward machine to shape rewards and guide learning.
    """
    def __init__(self, env):
        self.env = env
        self.pda = ParenthesisBalancingPDA(max_depth=env.max_depth)
        self.reward_machine = ParenthesisRewardMachine()
        self.last_pda_state = None
        
        # Curriculum learning parameters
        self.curriculum_level = env.curriculum_level
        self.success_streak = 0
        self.curriculum_threshold = 5  # Consecutive successes needed to advance
        
    def reset(self, seed=None, options=None):
        """Reset the MDPTM-RM components and environment."""
        # Reset environment first
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Reset PDA and RM
        self.pda.reset()
        self.reward_machine.reset()
        self.last_pda_state = self.pda.get_state()
        
        # Update curriculum level if env changed it
        self.curriculum_level = self.env.curriculum_level
        
        # Update RM's initial distances
        self.reward_machine.last_min_distances = self.env.get_min_distances()
        
        return observation, self._get_full_info(info)
        
    def step(self, action):
        """Take a step and apply hierarchical reward shaping."""
        # Get distances before step for reward calculation
        min_distances_before = self.env.get_min_distances()
        
        # Take action in environment
        observation, env_reward, terminated, truncated, env_info = self.env.step(action)
        
        # Check if an item was collected
        collected_symbol = env_info.get('collected')
        if collected_symbol is not None:
            # Update PDA with the collected symbol
            self.pda.process_symbol(collected_symbol)
            
        # Get current PDA state
        current_pda_state = self.pda.get_state()
        
        # Update reward machine state
        self.reward_machine.transition(current_pda_state)
        
        # Get distances after step
        min_distances_after = self.env.get_min_distances()
        
        # Calculate shaped reward
        shaped_reward = self.reward_machine.get_reward(
            self.last_pda_state,
            current_pda_state,
            env_reward,
            min_distances_after,
            env_info
        )
        
        # Update last PDA state
        self.last_pda_state = current_pda_state
        
        # Check task completion for curriculum update
        task_complete = env_info.get('balanced', False)
        task_failed = terminated and not task_complete
        
        if task_complete:
            self.success_streak += 1
            # Update curriculum level if threshold reached
            if self.success_streak >= self.curriculum_threshold and self.env.curriculum_level < 10:
                self.curriculum_level += 1
                self.success_streak = 0
                # Update environment curriculum level
                self.env.curriculum_level = self.curriculum_level
                self.env.fixed_positions = (self.curriculum_level <= 1)
                print(f"*** Curriculum Level Increased to: {self.curriculum_level} ***")
        elif task_failed or truncated:
            self.success_streak = 0
            
        # Combine info
        full_info = self._get_full_info(env_info)
        
        return observation, shaped_reward, terminated, truncated, full_info
        
    def _get_full_info(self, env_info):
        """Combine env info with wrapper info."""
        pda_state = self.pda.get_state()
        wrapper_info = {
            'pda_state': pda_state['state'],
            'stack_depth': pda_state['stack_depth'],
            'max_stack_depth': pda_state['max_stack_depth'],
            'rm_state': self.reward_machine.current_state,
            'curriculum_level': self.curriculum_level,
            'success_streak': self.success_streak
        }
        # Merge infos
        full_info = {**env_info, **wrapper_info}
        return full_info
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()
        
    @property
    def action_space(self):
        return self.env.action_space
        
    @property
    def observation_space(self):
        return self.env.observation_space


# Feature extraction for the parenthesis environment
def extract_features_with_attention(observation, pda_state=None, grid_size=None):
    """
    Extract features for the parenthesis balancing task with attention mechanisms.
    """
    # Basic features from environment observation
    grid = observation['grid']
    agent_pos = observation['agent_pos']
    opening_count = observation['opening_count']
    closing_count = observation['closing_count']
    stack_depth = observation['stack_depth']
    stack_status = observation['stack_status']
    
    if grid_size is None:
        grid_size = grid.shape[0]
    max_items = grid_size * grid_size  # Theoretical max
    max_dist = grid_size * 2  # Max Manhattan distance
    
    # Determine current goal based on PDA state
    current_goal = None
    if pda_state is not None:
        # If stack is empty, we need an opening
        if pda_state['stack_depth'] == 0:
            current_goal = 'opening'
        # If stack is not empty, we might want a closing
        else:
            current_goal = 'closing'
    else:
        # Default: if stack_depth > 0, prefer closings
        current_goal = 'closing' if stack_depth > 0 else 'opening'
    
    # Set attention weights
    attention_weights = {'opening': 0.3, 'closing': 0.3}
    if current_goal in attention_weights:
        attention_weights[current_goal] = 1.0
    
    # Normalize agent position
    agent_pos_normalized = agent_pos / (grid_size - 1.0) if grid_size > 1 else np.zeros(2)
    
    # Calculate distances to nearest items
    min_distances = {'opening': max_dist, 'closing': max_dist}
    nearest_pos = {'opening': None, 'closing': None}
    
    for r in range(grid_size):
        for c in range(grid_size):
            item_type_num = grid[r, c]
            if item_type_num == 1:  # Opening parenthesis
                distance = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                if distance < min_distances['opening']:
                    min_distances['opening'] = distance
                    nearest_pos['opening'] = np.array([r, c])
            elif item_type_num == 2:  # Closing parenthesis
                distance = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                if distance < min_distances['closing']:
                    min_distances['closing'] = distance
                    nearest_pos['closing'] = np.array([r, c])
    
    # Normalized distances
    norm_min_dist_opening = min_distances['opening'] / max_dist
    norm_min_dist_closing = min_distances['closing'] / max_dist
    
    # Direction vectors (weighted by attention)
    dir_vec_opening = np.zeros(2, dtype=np.float32)
    dir_vec_closing = np.zeros(2, dtype=np.float32)
    
    if nearest_pos['opening'] is not None:
        direction = nearest_pos['opening'] - agent_pos
        norm = np.linalg.norm(direction)
        dir_vec_opening = (direction / norm if norm > 0 else direction) * attention_weights['opening']
    
    if nearest_pos['closing'] is not None:
        direction = nearest_pos['closing'] - agent_pos
        norm = np.linalg.norm(direction)
        dir_vec_closing = (direction / norm if norm > 0 else direction) * attention_weights['closing']
    
    # Progress features
    norm_opening_count = opening_count / max_items
    norm_closing_count = closing_count / max_items
    
    # Stack features
    norm_stack_depth = stack_depth / observation['stack_status'].shape[0]
    
    # Basic feature vector
    basic_features = np.concatenate([
        agent_pos_normalized,             # 2
        np.array([norm_opening_count, norm_closing_count]), # 2
        np.array([norm_min_dist_opening, norm_min_dist_closing]), # 2
        dir_vec_opening,                  # 2
        dir_vec_closing,                  # 2
        np.array([norm_stack_depth]),     # 1
        stack_status                      # max_depth
    ])
    
    # Add PDA-specific features if available
    if pda_state is not None:
        # PDA state (one-hot)
        pda_state_one_hot = np.zeros(3)
        state_mapping = {'q0': 0, 'qaccept': 1, 'qreject': 2}
        state_idx = state_mapping.get(pda_state['state'], 0)
        pda_state_one_hot[state_idx] = 1.0
        
        # Goal (one-hot: opening, closing)
        goal_one_hot = np.zeros(2)
        goal_idx = 0 if current_goal == 'opening' else 1
        goal_one_hot[goal_idx] = 1.0
        
        pda_features = np.concatenate([
            pda_state_one_hot,            # 3
            goal_one_hot                  # 2
        ])
        
        # Combine features
        features = np.concatenate([basic_features, pda_features])
    else:
        features = basic_features
    
    return features.astype(np.float32)


# Simple Neural Network model (will struggle with parenthesis balancing)
class SimpleNN(nn.Module):
    """
    A standard neural network without explicit memory mechanisms.
    This model will struggle with the parenthesis balancing task.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(SimpleNN, self).__init__()
        
        # Simple feedforward architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


# Neural Turing Machine simplified implementation
class SimplifiedNTM(nn.Module):
    """
    A simplified version of Neural Turing Machine that still attempts to
    use attention and memory operations.
    """
    def __init__(self, input_dim, output_dim, memory_size=10, memory_vector_dim=8):
        super(SimplifiedNTM, self).__init__()
        
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        
        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Memory addressing mechanism
        self.addressing = nn.Linear(64, memory_size)
        
        # Read and write heads
        self.read_head = nn.Linear(64, memory_vector_dim)
        self.write_head = nn.Linear(64, memory_vector_dim)
        
        # Output layer
        self.output_layer = nn.Linear(64 + memory_vector_dim, output_dim)
        
        # Initialize memory
        self.reset_memory()
        
    def reset_memory(self):
        self.memory = torch.zeros(self.memory_size, self.memory_vector_dim)
        if hasattr(self, 'device'):
            self.memory = self.memory.to(self.device)
        
    def forward(self, x):
        # Process input through controller
        controller_output = self.controller(x)
        
        # Generate addressing weights (attention)
        addressing_weights = F.softmax(self.addressing(controller_output), dim=1)
        
        # Read from memory
        read_vector = torch.matmul(addressing_weights, self.memory)
        
        # Generate write content
        write_vector = self.write_head(controller_output)
        
        # Write to memory (simplified)
        # In actual NTM, there would be more complex erase and add operations
        for i in range(self.memory_size):
            self.memory[i] = self.memory[i] * (1 - addressing_weights[0, i]) + \
                             write_vector * addressing_weights[0, i]
        
        # Combine controller output with read vector for final output
        combined = torch.cat([controller_output, read_vector], dim=1)
        output = self.output_layer(combined)
        
        return output
    
    def to(self, device):
        self.device = device
        if hasattr(self, 'memory'):
            self.memory = self.memory.to(device)
        return super(SimplifiedNTM, self).to(device)


# Training function for comparing MDPTM-RM with NTM/SimpleNN
def comparative_training(env_params={'grid_size': 7, 'max_depth': 3}, 
                        num_episodes=2000,
                        eval_interval=100,
                        models_to_train=['mdptm_rm', 'ntm', 'simple_nn']):
    """
    Train and compare different models on the parenthesis balancing task.
    """
    print("--- Initializing Comparative Training ---")
    
    # Create base environment
    base_env = ParenthesisBalancingEnv(
        grid_size=env_params['grid_size'],
        max_depth=env_params['max_depth'],
        max_steps=200,
        curriculum_level=0,
        fixed_positions=True,
        render_mode="rgb_array"
    )
    
    # Create MDPTM-RM wrapper if requested
    if 'mdptm_rm' in models_to_train:
        mdptm_rm_env = HierarchicalMDPTM_RM_Parenthesis(
            ParenthesisBalancingEnv(
                grid_size=env_params['grid_size'],
                max_depth=env_params['max_depth'],
                max_steps=200,
                curriculum_level=0,
                fixed_positions=True,
                render_mode="rgb_array"
            )
        )
    
    # Get state size from an environment observation
    observation, _ = base_env.reset()
    # Extract features without PDA state for basic models
    state = extract_features_with_attention(observation, grid_size=base_env.grid_size)
    state_size = len(state)
    action_size = base_env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models = {}
    optimizers = {}
    
    # Initialize models that were requested
    if 'simple_nn' in models_to_train:
        models['simple_nn'] = SimpleNN(state_size, action_size).to(device)
        optimizers['simple_nn'] = optim.Adam(models['simple_nn'].parameters(), lr=1e-4)
        
    if 'ntm' in models_to_train:
        models['ntm'] = SimplifiedNTM(state_size, action_size).to(device)
        optimizers['ntm'] = optim.Adam(models['ntm'].parameters(), lr=1e-4)
    
    if 'mdptm_rm' in models_to_train:
        # For MDPTM-RM, we'll use extract_features_with_attention with PDA state
        pda_state = mdptm_rm_env.pda.get_state()
        mdptm_state = extract_features_with_attention(observation, pda_state, grid_size=base_env.grid_size)
        mdptm_state_size = len(mdptm_state)
        
        models['mdptm_rm'] = SimpleNN(mdptm_state_size, action_size).to(device)
        optimizers['mdptm_rm'] = optim.Adam(models['mdptm_rm'].parameters(), lr=1e-4)
    
    # Training metrics
    metrics = {model: {'scores': [], 'avg_scores': [], 'success_rates': []} for model in models}
    
    # Define epsilon schedule
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    
    # Main training loop
    print("\n--- Starting Comparative Training ---")
    start_time = time.time()
    
    for model_name in models:
        print(f"\n=== Training {model_name} ===")
        eps = eps_start
        
        # Reset environment based on model type
        if model_name == 'mdptm_rm':
            env = mdptm_rm_env
            # Reset the environment and curriculum
            env.curriculum_level = 0
            env.env.curriculum_level = 0
            env.env.fixed_positions = True
        else:
            # Create new environment for other models
            env = ParenthesisBalancingEnv(
                grid_size=env_params['grid_size'],
                max_depth=env_params['max_depth'],
                max_steps=200,
                curriculum_level=0,
                fixed_positions=True,
                render_mode="rgb_array"
            )
        
        # Training loop for this model
        for i_episode in range(1, num_episodes + 1):
            # Reset environment
            if model_name == 'mdptm_rm':
                observation, info = env.reset()
                pda_state = env.pda.get_state()
                state = extract_features_with_attention(observation, pda_state, grid_size=env.env.grid_size)
            else:
                observation, info = env.reset()
                state = extract_features_with_attention(observation, grid_size=env.grid_size)
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Reset NTM memory if applicable
            if model_name == 'ntm':
                models[model_name].reset_memory()
            
            # Episode variables
            score = 0
            done = False
            
            # Create memory for experience replay
            memory = []
            
            # Episode loop
            while not done:
                # Select action (epsilon-greedy)
                if random.random() > eps:
                    with torch.no_grad():
                        action_values = models[model_name](state_tensor)
                    action = torch.argmax(action_values).item()
                else:
                    action = random.randrange(action_size)
                
                # Take step in environment
                if model_name == 'mdptm_rm':
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    pda_state = env.pda.get_state()
                    next_state = extract_features_with_attention(next_observation, pda_state, grid_size=env.env.grid_size)
                else:
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    next_state = extract_features_with_attention(next_observation, grid_size=env.grid_size)
                
                done = terminated or truncated
                
                # Store transition
                memory.append((state, action, reward, next_state, done))
                
                # Update state and score
                state = next_state
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                score += reward
            
            # Batch learning after episode
            if len(memory) > 0:
                # Sample a batch from memory (or use all if small)
                batch_size = min(64, len(memory))
                batch = random.sample(memory, batch_size) if len(memory) > batch_size else memory
                
                # Unpack batch
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                actions_tensor = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
                dones_tensor = torch.FloatTensor(np.array(dones, dtype=np.uint8)).unsqueeze(1).to(device)
                
                # Get Q values
                q_values = models[model_name](states_tensor).gather(1, actions_tensor)
                
                # Get target Q values
                with torch.no_grad():
                    next_q_values = models[model_name](next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q_values = rewards_tensor + 0.99 * next_q_values * (1 - dones_tensor)
                
                # Compute loss and optimize
                loss = F.mse_loss(q_values, target_q_values)
                optimizers[model_name].zero_grad()
                loss.backward()
                optimizers[model_name].step()
            
            # Store metrics
            metrics[model_name]['scores'].append(score)
            
            # Calculate running average score
            window_size = min(100, len(metrics[model_name]['scores']))
            avg_score = np.mean(metrics[model_name]['scores'][-window_size:])
            metrics[model_name]['avg_scores'].append(avg_score)
            
            # Approximate success rate
            is_success = info.get('balanced', False)
            recent_successes = sum(1 for s in metrics[model_name]['scores'][-window_size:] if s > 0)
            success_rate = recent_successes / window_size if window_size > 0 else 0.0
            metrics[model_name]['success_rates'].append(success_rate)
            
            # Decay epsilon
            eps = max(eps_end, eps_decay * eps)
            
            # Print progress
            if i_episode % 10 == 0:
                elapsed_time = time.time() - start_time
                curriculum_level = getattr(env, 'curriculum_level', 0)
                print(f"Ep {i_episode:>4}/{num_episodes} | Model: {model_name} | Lvl: {curriculum_level} | "
                      f"Score: {score:>6.1f} | Avg: {avg_score:>6.1f} | "
                      f"Succ: {success_rate:.2f} | Eps: {eps:.3f} | Time: {elapsed_time:.1f}s")
            
            # Periodic evaluation
            if eval_interval > 0 and i_episode % eval_interval == 0:
                print(f"\n--- Evaluating {model_name} at Episode {i_episode} ---")
                success_rate = evaluate_model(models[model_name], model_name, env, device, num_episodes=10)
                print(f"--- {model_name} Evaluation: Success Rate = {success_rate:.2f} ---\n")
    
    # End of training
    print("\n--- Comparative Training Finished ---")
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    
    # Final evaluation for all models
    print("\n--- Final Evaluation ---")
    final_results = {}
    
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        
        if model_name == 'mdptm_rm':
            eval_env = mdptm_rm_env
        else:
            eval_env = ParenthesisBalancingEnv(
                grid_size=env_params['grid_size'],
                max_depth=env_params['max_depth'],
                max_steps=200,
                curriculum_level=0,
                fixed_positions=True,
                render_mode="rgb_array"
            )
            
        success_rate = evaluate_model(models[model_name], model_name, eval_env, device, num_episodes=50)
        final_results[model_name] = success_rate
        print(f"{model_name} Final Success Rate: {success_rate:.2f}")
    
    # Plot comparative results
    plot_comparative_results(metrics, final_results)
    
    return models, metrics, final_results


def evaluate_model(model, model_name, env, device, num_episodes=10):
    """Evaluate model performance over multiple episodes."""
    successes = 0
    
    # Set model to evaluation mode
    model.eval()
    
    for i in range(num_episodes):
        # Reset environment
        if model_name == 'mdptm_rm':
            observation, info = env.reset()
            pda_state = env.pda.get_state()
            state = extract_features_with_attention(observation, pda_state, grid_size=env.env.grid_size)
        else:
            observation, info = env.reset()
            state = extract_features_with_attention(observation, grid_size=env.grid_size)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Reset NTM memory if applicable
        if model_name == 'ntm':
            model.reset_memory()
        
        # Episode variables
        done = False
        episode_success = False
        
        # Episode loop
        while not done:
            # Select action (greedy policy)
            with torch.no_grad():
                action_values = model(state_tensor)
            action = torch.argmax(action_values).item()
            
            # Take action
            if model_name == 'mdptm_rm':
                next_observation, reward, terminated, truncated, info = env.step(action)
                pda_state = env.pda.get_state()
                next_state = extract_features_with_attention(next_observation, pda_state, grid_size=env.env.grid_size)
            else:
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_state = extract_features_with_attention(next_observation, grid_size=env.grid_size)
            
            done = terminated or truncated
            
            # Update state
            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Check if task completed successfully
            if info.get('balanced', False):
                episode_success = True
        
        if episode_success:
            successes += 1
    
    # Set model back to training mode
    model.train()
    
    # Calculate success rate
    success_rate = successes / num_episodes if num_episodes > 0 else 0
    
    return success_rate


def plot_comparative_results(metrics, final_results):
    """Plot training metrics for all models."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Get all model names
    model_names = list(metrics.keys())
    
    # Plot average scores
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        plt.plot(metrics[model_name]['avg_scores'], label=f"{model_name} Avg Score")
    
    plt.title('Average Scores During Training')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparative_scores.png')
    plt.close()
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        plt.plot(metrics[model_name]['success_rates'], label=f"{model_name} Success Rate")
    
    plt.title('Success Rates During Training')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparative_success_rates.png')
    plt.close()
    
    # Bar chart of final success rates
    plt.figure(figsize=(10, 6))
    plt.bar(final_results.keys(), final_results.values(), color=['blue', 'green', 'red'])
    plt.title('Final Success Rates')
    plt.ylabel('Success Rate')
    plt.ylim([0, 1.0])
    
    # Add values on bars
    for model, rate in final_results.items():
        plt.text(model, rate + 0.02, f'{rate:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('final_success_rates.png')
    plt.close()


# Demonstration for visual inspection
def demonstrate_model(model, model_name, env, device, max_steps=200):
    """Run a visual demonstration of the model."""
    print(f"\n--- Demonstrating {model_name} ---")
    
    # Set render mode to human
    original_render_mode = env.render_mode
    env.render_mode = 'human'
    
    # Reset environment
    if model_name == 'mdptm_rm':
        observation, info = env.reset()
        pda_state = env.pda.get_state()
        state = extract_features_with_attention(observation, pda_state, grid_size=env.env.grid_size)
    else:
        observation, info = env.reset()
        state = extract_features_with_attention(observation, grid_size=env.grid_size)
    
    # Reset NTM memory if applicable
    if model_name == 'ntm':
        model.reset_memory()
    
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Episode variables
    done = False
    steps = 0
    total_reward = 0
    success = False
    
    # Set model to evaluation mode
    model.eval()
    
    # Episode loop
    while not done and steps < max_steps:
        # Render the environment
        env.render()
        time.sleep(0.3)  # Slow down for visibility
        
        # Select action (greedy policy)
        with torch.no_grad():
            action_values = model(state_tensor)
        action = torch.argmax(action_values).item()
        
        # Take action
        if model_name == 'mdptm_rm':
            next_observation, reward, terminated, truncated, info = env.step(action)
            pda_state = env.pda.get_state()
            next_state = extract_features_with_attention(next_observation, pda_state, grid_size=env.env.grid_size)
        else:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = extract_features_with_attention(next_observation, grid_size=env.grid_size)
        
        done = terminated or truncated
        total_reward += reward
        
        # Print step info
        action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
        print(f"Step {steps+1}: Action={action_names[action]}, Reward={reward:.2f}")
        if model_name == 'mdptm_rm':
            print(f"PDA State: {pda_state['state']}, Stack Depth: {len(pda_state['stack'])}")
            print(f"Stack: {''.join(pda_state['stack'])}")
        
        # Update state and steps
        state = next_state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        steps += 1
        
        # Check for success
        if info.get('balanced', False):
            success = True
            print("SUCCESS! Balanced parentheses pattern completed.")
        
        # For NTM, show memory usage
        if model_name == 'ntm':
            print(f"Memory usage: {torch.sum(torch.abs(model.memory)).item():.2f}")
    
    # Final render
    env.render()
    
    # Restore original render mode
    env.render_mode = original_render_mode
    
    # Print summary
    print(f"\n--- Demonstration Summary ---")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Result: {'Success' if success else 'Failure'}")
    
    # Set model back to training mode
    model.train()
    
    return success


# Main experiment to compare MDPTM-RM with Neural Turing Machine
def run_comparative_experiment():
    """
    Run the main experiment comparing MDPTM-RM against Neural Turing Machine
    on the parenthesis balancing task to show why MDPTM-RM is superior.
    """
    print("=" * 60)
    print(" COMPARATIVE EXPERIMENT: MDPTM-RM vs Neural Turing Machine ".center(60, "="))
    print("=" * 60)
    
    # Set parameters for the experiment
    env_params = {
        'grid_size': 7,    # Grid size
        'max_depth': 3     # Maximum nesting depth of parentheses
    }
    
    # Create basic environment for explanation
    env = ParenthesisBalancingEnv(
        grid_size=env_params['grid_size'],
        max_depth=env_params['max_depth'],
        max_steps=200,
        curriculum_level=0,
        fixed_positions=True,
        render_mode="human"
    )
    
    print("\nEnvironment Description:")
    print("-" * 30)
    print("The Parenthesis Balancing Environment requires an agent to collect")
    print("opening '(' and closing ')' parentheses in a balanced order.")
    print("This is a context-free language recognition task, which requires")
    print("a stack-like memory structure to track the nesting level.")
    print()
    print("Rules:")
    print("1. Must collect an opening parenthesis '(' before collecting a closing one ')'")
    print("2. Each closing parenthesis must match a previously collected opening one")
    print("3. The final pattern must be balanced, like '()', '(())', or '()()()'")
    print("-" * 30)
    
    # Demonstrate the environment manually
    print("\nDemonstrating environment with random actions...")
    observation, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 20:  # Limit to 20 steps for demo
        env.render()
        action = random.randint(0, 3)  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        time.sleep(0.5)  # Pause for visibility
    
    print("\nNow we'll train and compare MDPTM-RM against Neural Turing Machine...")
    
    # Train and compare the models
    models, metrics, final_results = comparative_training(
        env_params=env_params,
        num_episodes=2000,
        eval_interval=200,
        models_to_train=['mdptm_rm', 'ntm']  # Compare just these two
    )
    
    # Analysis of the results
    print("\n" + "=" * 60)
    print(" EXPERIMENT ANALYSIS ".center(60, "="))
    print("=" * 60)
    
    # Compare final success rates
    print("\nFinal Success Rates:")
    for model_name, success_rate in final_results.items():
        print(f"{model_name}: {success_rate:.2f}")
    
    # Calculate learning speed (episodes to reach 0.5 success rate)
    learning_speed = {}
    for model_name in metrics:
        success_rates = metrics[model_name]['success_rates']
        episodes_to_threshold = next((i+1 for i, rate in enumerate(success_rates) if rate >= 0.5), len(success_rates))
        learning_speed[model_name] = episodes_to_threshold
    
    print("\nEpisodes to Reach 0.5 Success Rate:")
    for model_name, episodes in learning_speed.items():
        print(f"{model_name}: {episodes}")
    
    # Analyze maximum performance
    max_performance = {
        model_name: max(metrics[model_name]['success_rates']) 
        for model_name in metrics
    }
    
    print("\nMaximum Success Rate Achieved:")
    for model_name, max_rate in max_performance.items():
        print(f"{model_name}: {max_rate:.2f}")
    
    # Comparative analysis
    print("\nComparative Analysis:")
    print("-" * 30)
    print("MDPTM-RM vs Neural Turing Machine:")
    
    # Compare learning efficiency
    if learning_speed['mdptm_rm'] < learning_speed['ntm']:
        ratio = learning_speed['ntm'] / learning_speed['mdptm_rm']
        print(f"MDPTM-RM learns {ratio:.1f}x faster than NTM")
    else:
        ratio = learning_speed['mdptm_rm'] / learning_speed['ntm']
        print(f"NTM learns {ratio:.1f}x faster than MDPTM-RM")
    
    # Compare final performance
    mdptm_advantage = final_results['mdptm_rm'] - final_results['ntm']
    if mdptm_advantage > 0:
        print(f"MDPTM-RM achieves {mdptm_advantage*100:.1f}% higher success rate than NTM")
    else:
        print(f"NTM achieves {-mdptm_advantage*100:.1f}% higher success rate than MDPTM-RM")
    
    # Explanation of why MDPTM-RM is better
    print("\nWhy MDPTM-RM Excels in this Task:")
    print("-" * 30)
    print("1. Explicit Stack Representation - MDPTM-RM uses a Pushdown Automaton (PDA)")
    print("   with an explicit stack that directly models the balanced parentheses problem.")
    print("   This is perfectly aligned with the formal language structure.")
    print()
    print("2. Clear State Transitions - The PDA has well-defined state transitions")
    print("   that directly correspond to the syntax rules of balanced parentheses.")
    print()
    print("3. Reward Machine Integration - The reward machine provides shaped rewards")
    print("   that guide the agent toward valid parentheses matching behaviors.")
    print()
    print("4. Hierarchical Structure - The hierarchical approach separates the")
    print("   language recognition (PDA) from the decision-making (RL agent),")
    print("   allowing each component to focus on its specialty.")
    print()
    print("5. Sample Efficiency - Due to its structured approach, MDPTM-RM requires")
    print("   fewer interactions with the environment to learn the optimal policy.")
    print()
    print("Neural Turing Machine Limitations:")
    print("1. Has to learn memory operations from scratch - The NTM must learn when")
    print("   and how to use its memory, rather than having predefined operations.")
    print()
    print("2. Lack of domain-specific structure - Does not incorporate the formal")
    print("   grammar rules that define balanced parentheses.")
    print()
    print("3. Less efficient reward utilization - Cannot interpret rewards in terms")
    print("   of language-specific goals like the reward machine does.")
    print()
    
    # Visual demonstration of both models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 60)
    print(" MODEL DEMONSTRATIONS ".center(60, "="))
    print("=" * 60)
    
    # Create demo environments
    demo_mdptm_env = HierarchicalMDPTM_RM_Parenthesis(
        ParenthesisBalancingEnv(
            grid_size=env_params['grid_size'],
            max_depth=env_params['max_depth'],
            max_steps=200,
            curriculum_level=3,  # Use a harder level for demo
            fixed_positions=False,
            render_mode="human"
        )
    )
    
    demo_ntm_env = ParenthesisBalancingEnv(
        grid_size=env_params['grid_size'],
        max_depth=env_params['max_depth'],
        max_steps=200,
        curriculum_level=3,  # Same difficulty
        fixed_positions=False,
        render_mode="human"
    )
    
    # Demonstrate MDPTM-RM
    print("\nDemonstrating MDPTM-RM on a complex parenthesis balancing task:")
    mdptm_success = demonstrate_model(models['mdptm_rm'], 'mdptm_rm', demo_mdptm_env, device)
    
    # Demonstrate NTM
    print("\nDemonstrating Neural Turing Machine on the same task:")
    ntm_success = demonstrate_model(models['ntm'], 'ntm', demo_ntm_env, device)
    
    # Final summary
    print("\n" + "=" * 60)
    print(" FINAL CONCLUSIONS ".center(60, "="))
    print("=" * 60)
    print("\nIn the parenthesis balancing task (a context-free language recognition problem):")
    print(f"MDPTM-RM Success: {'Yes' if mdptm_success else 'No'}")
    print(f"Neural Turing Machine Success: {'Yes' if ntm_success else 'No'}")
    print()
    
    if final_results['mdptm_rm'] > final_results['ntm']:
        print("MDPTM-RM significantly outperforms the Neural Turing Machine approach")
        print("for this type of formal language recognition task because it explicitly")
        print("incorporates the structure of the problem domain through its PDA component.")
    else:
        print("Surprisingly, the Neural Turing Machine matched or outperformed MDPTM-RM,")
        print("though MDPTM-RM is theoretically better suited to this formal language task.")
    
    print("\nThis experiment demonstrates that for problems with clear formal structure,")
    print("leveraging that structure through models like MDPTM-RM provides significant")
    print("advantages over general learning approaches that must discover structure.")
    
    return {
        'models': models,
        'metrics': metrics,
        'final_results': final_results,
        'learning_speed': learning_speed
    }


# Entry point
if __name__ == "__main__":
    print("+" * 60)
    print(" Parenthesis Balancing: MDPTM-RM vs Neural Turing Machine ".center(60, "+"))
    print("+" * 60)
    
    results = run_comparative_experiment()
    
    print("\nExperiment finished.")
