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
# Note: IPython.display.clear_output is specific to Jupyter/IPython environments.
# If running as a standard script, this might need adjustment or removal.
try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False):
        # Simple fallback for non-IPython environments
        print("\033c", end="") # ANSI escape code to clear screen

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) # For GPU reproducibility

# Simplified version of the TreasureHuntEnv
class SimplifiedTreasureHuntEnv(gym.Env):
    """
    A simplified grid-based environment where an agent must collect symbols 'a', 'b', and 'c'
    in the pattern a^n b^n c^n for some n determined at the start of each episode.

    Simplifications:
    - Smaller grid size
    - Option for fixed n value
    - Option for fixed symbol positions
    - Clear visual indicators
    - Progressive difficulty levels
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, grid_size=6, max_n=2, max_steps=100, fixed_n=None,
                 fixed_positions=False, curriculum_level=0, render_mode='human'):
        super(SimplifiedTreasureHuntEnv, self).__init__()

        self.grid_size = grid_size
        self.max_n = max_n  # Maximum value of n for a^n b^n c^n
        self.max_steps = max_steps
        self.fixed_n = fixed_n  # If not None, use this fixed value of n
        self.fixed_positions = fixed_positions  # If True, use fixed positions for symbols
        self.curriculum_level = curriculum_level  # 0: easiest, higher = harder
        self.render_mode = render_mode # Added render_mode

        # Define the action space (0: up, 1: right, 2: down, 3: left)
        self.action_space = spaces.Discrete(4)

        # Define the observation space
        # Grid + agent position + remaining items + current phase
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32),
            'agent_pos': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            'remaining_a': spaces.Discrete(max_n+1),
            'remaining_b': spaces.Discrete(max_n+1),
            'remaining_c': spaces.Discrete(max_n+1),
            'current_phase': spaces.Discrete(4)  # 0: initial/a, 1: b, 2: c, 3: done
        })

        # Initialize the environment state
        self.grid = None
        self.agent_pos = None
        self.remaining_items = None
        self.current_step = 0
        self.n_value = None
        self.current_phase = None  # 0: collecting a, 1: collecting b, 2: collecting c, 3: done

        # Item type encoding: 0 is empty, 1 is 'a', 2 is 'b', 3 is 'c'
        self.item_types = {0: 'empty', 1: 'a', 2: 'b', 3: 'c'}
        self.item_colors = {0: 'white', 1: 'red', 2: 'green', 3: 'blue'}

        # For tracking progress
        self.collected_sequence = []

        # For rendering
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Comply with newer gym API
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize an empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Determine the value of n for this episode
        if self.fixed_n is not None:
            self.n_value = self.fixed_n
        else:
            # For curriculum learning, adjust n based on level
            if self.curriculum_level == 0:
                self.n_value = 1  # Start with simplest case
            else:
                # Increase n gradually, up to max_n
                self.n_value = min(1 + self.curriculum_level // 2, self.max_n)

        # Initialize remaining items
        self.remaining_items = {'a': self.n_value, 'b': self.n_value, 'c': self.n_value}

        # Start in the 'a' collection phase
        self.current_phase = 0

        # Reset step counter and collected sequence
        self.current_step = 0
        self.collected_sequence = []

        # Place the agent at a specific position based on curriculum level
        if self.curriculum_level <= 1 or self.fixed_positions:
             # Fixed starting position for easier learning (consistent)
            self.agent_pos = np.array([self.grid_size -1, self.grid_size // 2])
        else:
            # Random position for harder levels
            valid_start = False
            while not valid_start:
                self.agent_pos = self.np_random.integers(0, self.grid_size, size=2, dtype=np.int32)
                # Ensure agent doesn't start on an item (items placed later, but good practice)
                if self.grid[self.agent_pos[0], self.agent_pos[1]] == 0:
                    valid_start = True

        # Place items on the grid
        self._place_items()

        # Return initial observation and info dict
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_items(self):
        """Place 'a', 'b', and 'c' items on the grid with curriculum-based positioning."""
        # Clear any existing items
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        if self.fixed_positions and self.curriculum_level <= 1:
            # Level 0-1: Fixed positions in distinct regions for easier learning
            # 'a' items in top-left quadrant
            self._place_items_in_region(1, self.n_value, 0, self.grid_size//2 - 1, 0, self.grid_size//2 - 1)
            # 'b' items in top-right quadrant
            self._place_items_in_region(2, self.n_value, 0, self.grid_size//2 - 1, self.grid_size//2, self.grid_size - 1)
            # 'c' items in bottom-right quadrant (near agent start if fixed)
            self._place_items_in_region(3, self.n_value, self.grid_size//2, self.grid_size - 1, self.grid_size//2, self.grid_size - 1)
        elif self.curriculum_level <= 3:
            # Level 2-3: Semi-structured placement with some randomization
            self._place_items_with_distance(1, self.n_value, min_distance=1)  # 'a' items
            self._place_items_with_distance(2, self.n_value, min_distance=1)  # 'b' items
            self._place_items_with_distance(3, self.n_value, min_distance=1)  # 'c' items
        else:
            # Level 4+: Fully random placement
            self._place_random_items(1, self.n_value)  # 'a' items
            self._place_random_items(2, self.n_value)  # 'b' items
            self._place_random_items(3, self.n_value)  # 'c' items

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
             # Fallback if region is too small or agent blocks too many spots
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
        max_attempts_per_item = 100 # Increase attempts
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
                        total_attempts = 0 # Reset attempts after successful placement

        # If we couldn't place all items with distance constraints, fall back to random placement
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
             print(f"Error: Could not place all {count} items of type {item_type} randomly after {max_attempts} attempts. Grid might be too full.")


    def _get_observation(self):
        """Return the current state as an observation."""
        return {
            'grid': self.grid.copy(),
            'agent_pos': self.agent_pos.copy(),
            'remaining_a': self.remaining_items['a'],
            'remaining_b': self.remaining_items['b'],
            'remaining_c': self.remaining_items['c'],
            'current_phase': self.current_phase
        }

    def _get_info(self):
        """Return auxiliary information about the state."""
        min_distances = self.get_min_distances()
        return {
            'n_value': self.n_value,
            'collected_sequence': self.collected_sequence.copy(),
            'min_dist_a': min_distances['a'],
            'min_dist_b': min_distances['b'],
            'min_dist_c': min_distances['c']
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
        info = {'collected': None, 'phase_complete': False, 'task_complete': False, 'error': None} # Added 'error' key

        # Process item collection if there's an item
        if item_at_pos > 0:
            item_type = self.item_types[item_at_pos]

            # Check if the collected item is of the correct type for the current phase
            expected_item = ['a', 'b', 'c'][self.current_phase]

            if item_type == expected_item:
                # Correct item for the current phase
                self.remaining_items[item_type] -= 1
                self.collected_sequence.append(item_type)
                reward += 2.0  # Better reward for collecting correct item
                info['collected'] = item_type

                # Remove the item from the grid
                self.grid[x, y] = 0

                # Check if we've completed the current phase
                if self.remaining_items[item_type] == 0:
                    reward += 10.0  # Larger bonus for completing a phase
                    info['phase_complete'] = True
                    self.current_phase += 1

                    # Check if we've completed all phases (collected a^n b^n c^n)
                    if self.current_phase == 3:
                        reward += 50.0  # Large bonus for completing the pattern
                        terminated = True
                        info['task_complete'] = True
            else:
                # Wrong item for current phase - penalty
                reward -= 5.0
                terminated = True  # End episode on wrong collection
                info['error'] = f"Collected {item_type} during phase {self.current_phase} (expected {expected_item})"

        # Check if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated: # Only add max steps error if not already terminated
                info['error'] = "Max steps reached"

        # Get observation and full info dict
        observation = self._get_observation()
        full_info = self._get_info()
        full_info.update(info) # Merge step-specific info

        if self.render_mode == "human":
            self._render_frame()

        # Return the step information
        return observation, reward, terminated, truncated, full_info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
             self._render_frame() # Let _render_frame handle display for human mode

    def _render_frame(self):
        # Basic text rendering fallback if Pygame not available or human mode
        if self.render_mode == "human":
            clear_output(wait=True) # Use the imported clear_output
            print(f"Step: {self.current_step}/{self.max_steps} | Phase: {self.current_phase} | n={self.n_value}")
            print(f"Remaining: a={self.remaining_items['a']}, b={self.remaining_items['b']}, c={self.remaining_items['c']}")
            print(f"Collected: {''.join(self.collected_sequence)}")
            grid_repr = ""
            for r in range(self.grid_size):
                row_str = "|"
                for c in range(self.grid_size):
                    if r == self.agent_pos[0] and c == self.agent_pos[1]:
                        row_str += " A "  # Agent
                    elif self.grid[r, c] == 1:
                        row_str += " a "
                    elif self.grid[r, c] == 2:
                        row_str += " b "
                    elif self.grid[r, c] == 3:
                        row_str += " c "
                    else:
                        row_str += " . "
                grid_repr += row_str + "|\n"
            print("-" * (self.grid_size * 3 + 2))
            print(grid_repr, end="")
            print("-" * (self.grid_size * 3 + 2))
            time.sleep(0.1) # Add a small delay for visibility
            return None # Human mode doesn't return an array

        # Fallback for rgb_array (can be improved with Pygame later)
        elif self.render_mode == "rgb_array":
             # Create a simple RGB array representation
             rgb_array = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
             # Background: Gray
             rgb_array[:,:,:] = 128

             # Items
             for r in range(self.grid_size):
                 for c in range(self.grid_size):
                     if self.grid[r, c] == 1: # 'a': Red
                         rgb_array[r, c] = [255, 0, 0]
                     elif self.grid[r, c] == 2: # 'b': Green
                         rgb_array[r, c] = [0, 255, 0]
                     elif self.grid[r, c] == 3: # 'c': Blue
                         rgb_array[r, c] = [0, 0, 255]

             # Agent: White
             ax, ay = self.agent_pos
             if 0 <= ax < self.grid_size and 0 <= ay < self.grid_size:
                 rgb_array[ax, ay] = [255, 255, 255]

             # Upscale for better visibility if needed
             scale = 50
             rgb_array = np.kron(rgb_array, np.ones((scale, scale, 1), dtype=np.uint8))
             return rgb_array


    def close(self):
        # Clean up any rendering resources if using Pygame
        pass # No Pygame resources to close in this version

    def get_min_distances(self):
        """Calculate minimum Manhattan distance from agent to each item type."""
        min_distances = {'a': float('inf'), 'b': float('inf'), 'c': float('inf')}
        agent_r, agent_c = self.agent_pos

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                item_type_num = self.grid[r, c]
                if item_type_num > 0:
                    item_type = self.item_types[item_type_num]
                    distance = abs(r - agent_r) + abs(c - agent_c)
                    min_distances[item_type] = min(min_distances[item_type], distance)

        # Replace infinite distances with a large number (e.g., grid diagonal)
        # This prevents issues with calculations later if an item type is gone.
        max_dist = self.grid_size * 2
        for item_type in min_distances:
            if min_distances[item_type] == float('inf'):
                min_distances[item_type] = max_dist

        return min_distances

# Hierarchical TuringMachine with improved phase management
class HierarchicalTuringMachine:
    """
    An enhanced Turing Machine for the a^n b^n c^n language recognition with
    hierarchical structure and improved state tracking.
    """
    def __init__(self, n_value=0): # Accept n_value at init
        # Initialize states
        self.states = ['q0', 'qa', 'qb', 'qc', 'qaccept', 'qreject']
        self.current_state = 'q0'

        # Store the target n for the episode
        self.target_n = n_value

        # Initialize tape with blank symbol '#' (less critical here, focus on counts)
        self.tape = ['#']
        self.head_position = 0

        # Initialize counters
        self.a_count = 0
        self.b_count = 0
        self.c_count = 0

        # Initialize phase tracker (match env phases)
        self.phase = 0  # 0: collecting a, 1: collecting b, 2: collecting c, 3: accepted

        # Phase completion tracking (more detailed)
        self.phase_complete = {'a': False, 'b': False, 'c': False}

        # For hierarchical decision making
        self.current_goal = 'a'  # Current symbol we're trying to collect

    def reset(self, n_value): # Accept n_value on reset
        """Reset the Turing Machine."""
        self.current_state = 'q0'
        self.target_n = n_value # Update target n
        self.tape = ['#']
        self.head_position = 0
        self.a_count = 0
        self.b_count = 0
        self.c_count = 0
        self.phase = 0
        self.phase_complete = {'a': False, 'b': False, 'c': False}
        self.current_goal = 'a'

    def process_symbol(self, symbol):
        """Process a symbol input to the Turing Machine based on counts and target n."""
        if self.current_state in ['qaccept', 'qreject']:
            return self.current_state

        # --- State transitions based on counts and phase ---
        if symbol == 'a':
            if self.phase == 0: # Expecting 'a's
                self.a_count += 1
                self.current_state = 'qa'
                self.current_goal = 'a' # Still need 'a's unless count is met
                # Write 'A' to tape (optional visualization)
                self._write_tape('A')

                if self.a_count == self.target_n:
                    self.phase_complete['a'] = True
                    self.phase = 1 # Move to 'b' phase
                    self.current_goal = 'b' # Now look for 'b's
            else:
                self.current_state = 'qreject' # 'a' received out of order

        elif symbol == 'b':
            if self.phase == 1: # Expecting 'b's after finishing 'a's
                self.b_count += 1
                self.current_state = 'qb'
                self.current_goal = 'b' # Still need 'b's unless count is met
                self._write_tape('B')

                if self.b_count > self.target_n: # Too many 'b's
                    self.current_state = 'qreject'
                elif self.b_count == self.target_n:
                    self.phase_complete['b'] = True
                    self.phase = 2 # Move to 'c' phase
                    self.current_goal = 'c' # Now look for 'c's
            else:
                 self.current_state = 'qreject' # 'b' received out of order

        elif symbol == 'c':
            if self.phase == 2: # Expecting 'c's after finishing 'b's
                self.c_count += 1
                self.current_state = 'qc'
                self.current_goal = 'c' # Still need 'c's unless count is met
                self._write_tape('C')

                if self.c_count > self.target_n: # Too many 'c's
                    self.current_state = 'qreject'
                elif self.c_count == self.target_n:
                     # All counts match target n, accept
                     self.phase_complete['c'] = True
                     self.current_state = 'qaccept'
                     self.phase = 3 # Acceptance phase
                     self.current_goal = None
            else:
                 self.current_state = 'qreject' # 'c' received out of order

        return self.current_state

    def _write_tape(self, char):
        """Helper to write to tape (optional)."""
        if self.head_position >= len(self.tape):
            self.tape.append(char)
        else:
            self.tape[self.head_position] = char
        self.head_position += 1


    def get_state(self):
        """Return the current state of the Turing Machine."""
        return {
            'state': self.current_state,
            'tape': self.tape.copy(),
            'head_position': self.head_position,
            'a_count': self.a_count,
            'b_count': self.b_count,
            'c_count': self.c_count,
            'target_n': self.target_n, # Include target n
            'phase': self.phase,
            'current_goal': self.current_goal,
            'phase_complete': self.phase_complete.copy()
        }

    def set_goal(self, goal):
        """Allow external setting of the current goal (less relevant with count-based logic)."""
        if goal in ['a', 'b', 'c']:
            self.current_goal = goal
            return True
        return False

# Enhanced reward machine with improved distance-based shaping
class EnhancedRewardMachine:
    """
    An enhanced reward machine with more sophisticated shaping rewards
    to guide the agent toward the correct sequence.
    """
    def __init__(self):
        # Define states (simplified, align with TM phases)
        self.states = ['u_init', 'u_a', 'u_b', 'u_c', 'u_accept', 'u_reject']
        self.current_state = 'u_init'

        # Define reward shaping parameters (can be tuned)
        self.phase_completion_bonus = 10.0
        self.correct_item_reward = 2.0
        self.wrong_item_penalty = -5.0
        self.task_completion_bonus = 50.0
        self.step_penalty = -0.1

        # Distance-based shaping parameters
        self.distance_improvement_factor = 0.3 # Reduced factor
        self.distance_deterioration_factor = -0.2 # Reduced factor

        # Progress tracking for reward shaping
        self.last_min_distances = {'a': float('inf'), 'b': float('inf'), 'c': float('inf')}
        self.approach_counts = {'a': 0, 'b': 0, 'c': 0}  # Count steps moving toward each item type

    def reset(self):
        """Reset the reward machine."""
        self.current_state = 'u_init'
        self.last_min_distances = {'a': float('inf'), 'b': float('inf'), 'c': float('inf')}
        self.approach_counts = {'a': 0, 'b': 0, 'c': 0}

    def transition(self, tm_state):
        """Update the reward machine state based on the Turing Machine state/phase."""
        tm_s = tm_state['state']
        tm_phase = tm_state['phase']

        if tm_s == 'qaccept':
            self.current_state = 'u_accept'
        elif tm_s == 'qreject':
            self.current_state = 'u_reject'
        elif tm_phase == 0:
            self.current_state = 'u_a' # Or 'u_init' if more precise state needed
        elif tm_phase == 1:
            self.current_state = 'u_b'
        elif tm_phase == 2:
            self.current_state = 'u_c'
        else: # Should ideally not happen if phase/state are sync'd
             self.current_state = 'u_init'

        return self.current_state

    def get_reward(self, old_tm_state, new_tm_state, env_reward, distances=None, info=None):
        """Calculate shaped reward based on various factors."""
        # Start with the environment reward (which already includes penalties/bonuses for collection)
        final_reward = env_reward

        # Add distance-based shaping if distances provided
        if distances is not None:
            current_goal = new_tm_state['current_goal']
            if current_goal in distances:
                current_distance = distances[current_goal]
                prev_distance = self.last_min_distances[current_goal]

                # --- START CHANGE ---
                # Only calculate distance shaping reward if the previous distance was valid (not inf)
                # and the goal item still exists (distance < max_dist)
                if prev_distance != float('inf') and current_distance < self.last_min_distances.get(current_goal, float('inf')):
                    # Reward for getting closer to target
                    if current_distance < prev_distance:
                        improvement = prev_distance - current_distance
                        distance_reward = self.distance_improvement_factor * improvement
                        final_reward += distance_reward
                        self.approach_counts[current_goal] = min(5, self.approach_counts[current_goal] + 1) # Cap approach count
                    # Penalty for moving away from target (only if not just collected)
                    elif current_distance > prev_distance and not (info and info.get('collected') == current_goal):
                        deterioration = current_distance - prev_distance
                        distance_penalty = self.distance_deterioration_factor * deterioration
                        final_reward += distance_penalty
                        self.approach_counts[current_goal] = max(0, self.approach_counts[current_goal] - 1)
                    # else: distance unchanged, no shaping reward/penalty
                elif current_distance == prev_distance:
                     # Reset approach count if distance stagnates? Optional.
                     # self.approach_counts[current_goal] = 0
                     pass
                # --- END CHANGE ---

            # Update last distances *after* potential calculations
            self.last_min_distances = distances.copy()

        # Add consistency bonus (small reward for maintaining approach)
        if new_tm_state['current_goal'] in self.approach_counts and self.approach_counts[new_tm_state['current_goal']] > 2:
             consistency_bonus = 0.1 * self.approach_counts[new_tm_state['current_goal']]
             final_reward += min(0.5, consistency_bonus) # Cap bonus

        # Additional bonus for completing the pattern (already partially in env_reward)
        # Can add extra scaling based on n if desired
        if new_tm_state['state'] == 'qaccept' and old_tm_state['state'] != 'qaccept':
            n_value = new_tm_state['target_n']
            pattern_bonus = n_value * 2.0 # Small extra bonus based on n
            final_reward += pattern_bonus

        return final_reward


# Hierarchical MDPTM-RM with curriculum learning
class HierarchicalMDPTM_RM:
    """
    A hierarchical implementation of MDPTM-RM with curriculum learning
    and improved reward shaping. Acts as a wrapper around the environment.
    """
    def __init__(self, env):
        self.env = env
        # Initialize TM with n=0 initially, will be set in reset
        self.turing_machine = HierarchicalTuringMachine(n_value=0)
        self.reward_machine = EnhancedRewardMachine()
        self.last_tm_state = None
        self.collected_sequence = [] # Redundant? Env tracks this.

        # Curriculum learning parameters
        self.curriculum_level = env.curriculum_level # Sync with env
        self.success_streak = 0
        self.curriculum_threshold = 5 # Consecutive successes needed to advance

    def reset(self, seed=None, options=None):
        """Reset the MDPTM-RM components and the environment."""
        # Reset environment first to get n_value
        observation, info = self.env.reset(seed=seed, options=options)
        n_value = self.env.n_value # Get n from the reset env

        # Reset TM and RM
        self.turing_machine.reset(n_value=n_value)
        self.reward_machine.reset()
        self.last_tm_state = self.turing_machine.get_state()
        self.collected_sequence = [] # Reset sequence tracking

        # Update internal curriculum level if env changed it
        self.curriculum_level = self.env.curriculum_level

        # Update RM's initial distances
        self.reward_machine.last_min_distances = self.env.get_min_distances()

        return observation, self._get_full_info(info) # Return obs and combined info

    def step(self, action):
        """Take a step and apply hierarchical reward shaping."""
        # Get current distances *before* the step for RM calculation later
        min_distances_before = self.env.get_min_distances()

        # Take action in environment
        observation, env_reward, terminated, truncated, env_info = self.env.step(action)

        # Check if an item was collected from env_info
        collected_symbol = env_info.get('collected')
        if collected_symbol is not None:
            self.collected_sequence.append(collected_symbol) # Update sequence
            # Update Turing Machine
            self.turing_machine.process_symbol(collected_symbol)

        # Get current TM state
        current_tm_state = self.turing_machine.get_state()

        # Update the reward machine state (less critical for reward calc now)
        self.reward_machine.transition(current_tm_state)

        # Get distances *after* the step
        min_distances_after = self.env.get_min_distances() # Use current distances

        # Calculate shaped reward using RM logic
        shaped_reward = self.reward_machine.get_reward(
            self.last_tm_state,
            current_tm_state,
            env_reward,
            min_distances_after, # Use distances *after* the step for shaping
            env_info # Pass env_info which contains 'collected' etc.
        )

        # Update last TM state
        self.last_tm_state = current_tm_state

        # Check task completion status for curriculum update
        task_complete = env_info.get('task_complete', False)
        task_failed = terminated and not task_complete # Failed if terminated without success

        if task_complete:
            self.success_streak += 1
            # Update curriculum level if threshold reached
            if self.success_streak >= self.curriculum_threshold and self.env.curriculum_level < 10: # Cap level increase
                self.curriculum_level += 1
                self.success_streak = 0
                # Update environment curriculum level
                self.env.curriculum_level = self.curriculum_level
                self.env.fixed_positions = (self.curriculum_level <= 1) # Update fixed pos based on new level
                print(f"*** Curriculum Level Increased to: {self.curriculum_level} ***")
        elif task_failed or truncated: # Reset streak on failure or truncation
            self.success_streak = 0

        # Combine info
        full_info = self._get_full_info(env_info)

        return observation, shaped_reward, terminated, truncated, full_info

    def _get_full_info(self, env_info):
        """ Combine env info with wrapper info """
        tm_state = self.turing_machine.get_state()
        wrapper_info = {
            'tm_state_val': tm_state['state'], # Use string value
            'tm_phase': tm_state['phase'],
            'tm_goal': tm_state['current_goal'],
            'tm_a': tm_state['a_count'],
            'tm_b': tm_state['b_count'],
            'tm_c': tm_state['c_count'],
            'rm_state': self.reward_machine.current_state,
            'curriculum_level': self.curriculum_level,
            'success_streak': self.success_streak
        }
        # Merge env_info and wrapper_info, wrapper_info overrides if keys conflict
        full_info = {**env_info, **wrapper_info}
        return full_info


    def render(self, mode='human'):
         return self.env.render() # Delegate rendering to env

    def close(self):
         self.env.close() # Delegate closing to env

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        # This might need adjustment if the agent expects TM state in obs
        return self.env.observation_space


# Feature extraction with attention mechanisms
def extract_features_with_attention(observation, tm_state=None, grid_size=None): # Added grid_size
    """
    Extract a comprehensive feature vector with attention mechanisms to
    focus on relevant aspects of the state.
    """
    # Basic features from the environment observation dict
    grid = observation['grid']
    agent_pos = observation['agent_pos']
    remaining_a = observation['remaining_a']
    remaining_b = observation['remaining_b']
    remaining_c = observation['remaining_c']
    env_phase = observation['current_phase'] # Renamed to avoid clash with TM phase

    if grid_size is None:
        grid_size = grid.shape[0]
    max_items_per_type = grid_size * grid_size # Theoretical max
    max_dist = grid_size * 2 # Max Manhattan distance


    # --- Attention Mechanism ---
    # Determine current goal based on TM state if available, else env phase
    current_goal = None
    if tm_state is not None and tm_state['current_goal'] is not None:
        current_goal = tm_state['current_goal']
    else:
        phase_to_goal = {0: 'a', 1: 'b', 2: 'c'}
        current_goal = phase_to_goal.get(env_phase) # Get goal from env phase

    # Set attention weights: higher for the current goal
    attention_weights = {'a': 0.3, 'b': 0.3, 'c': 0.3}
    if current_goal in attention_weights:
        attention_weights[current_goal] = 1.0

    # --- Spatial Features ---
    # Agent position (normalized)
    agent_pos_normalized = agent_pos / (grid_size - 1.0) if grid_size > 1 else np.zeros(2)

    # Distances and directions to *nearest* item of each type
    min_distances = {'a': max_dist, 'b': max_dist, 'c': max_dist}
    nearest_pos = {'a': None, 'b': None, 'c': None}

    for r in range(grid_size):
        for c in range(grid_size):
            item_type_num = grid[r, c]
            if item_type_num > 0:
                item_char = {1: 'a', 2: 'b', 3: 'c'}[item_type_num]
                distance = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                if distance < min_distances[item_char]:
                    min_distances[item_char] = distance
                    nearest_pos[item_char] = np.array([r, c])

    # Normalized distances
    norm_min_dist_a = min_distances['a'] / max_dist
    norm_min_dist_b = min_distances['b'] / max_dist
    norm_min_dist_c = min_distances['c'] / max_dist

    # Normalized direction vectors (weighted by attention)
    dir_vec_a = np.zeros(2, dtype=np.float32)
    dir_vec_b = np.zeros(2, dtype=np.float32)
    dir_vec_c = np.zeros(2, dtype=np.float32)

    if nearest_pos['a'] is not None:
        direction = nearest_pos['a'] - agent_pos
        norm = np.linalg.norm(direction)
        dir_vec_a = (direction / norm if norm > 0 else direction) * attention_weights['a']
    if nearest_pos['b'] is not None:
        direction = nearest_pos['b'] - agent_pos
        norm = np.linalg.norm(direction)
        dir_vec_b = (direction / norm if norm > 0 else direction) * attention_weights['b']
    if nearest_pos['c'] is not None:
        direction = nearest_pos['c'] - agent_pos
        norm = np.linalg.norm(direction)
        dir_vec_c = (direction / norm if norm > 0 else direction) * attention_weights['c']

    # --- Task Progress Features ---
    # Remaining items (normalized)
    norm_rem_a = remaining_a / max_items_per_type
    norm_rem_b = remaining_b / max_items_per_type
    norm_rem_c = remaining_c / max_items_per_type

    # Environment phase (one-hot or normalized)
    env_phase_feature = np.zeros(4)
    env_phase_feature[min(env_phase, 3)] = 1.0 # One-hot encode env phase 0, 1, 2, 3

    # --- Basic Feature Vector ---
    basic_features = np.concatenate([
        agent_pos_normalized,          # 2
        np.array([norm_rem_a, norm_rem_b, norm_rem_c]), # 3
        np.array([norm_min_dist_a, norm_min_dist_b, norm_min_dist_c]), # 3
        dir_vec_a,                     # 2
        dir_vec_b,                     # 2
        dir_vec_c,                     # 2
        env_phase_feature              # 4
    ]) # Total basic: 18 features

    # --- Add TM-specific features if available ---
    if tm_state is not None:
        # TM state (one-hot)
        tm_state_one_hot = np.zeros(6)
        state_mapping = {'q0': 0, 'qa': 1, 'qb': 2, 'qc': 3, 'qaccept': 4, 'qreject': 5}
        state_idx = state_mapping.get(tm_state['state'], 0) # Default to q0 if unknown
        tm_state_one_hot[state_idx] = 1.0

        # TM phase (one-hot)
        tm_phase_one_hot = np.zeros(5) # 0, 1, 2, 3(accept), 4(reject?) - adjust if needed
        tm_phase_one_hot[min(tm_state['phase'], 4)] = 1.0 # Assuming phases 0, 1, 2, 3=accept, maybe add 4 for reject?

        # TM goal (one-hot: a, b, c, none)
        goal_one_hot = np.zeros(4)
        if tm_state['current_goal'] == 'a': goal_one_hot[0] = 1.0
        elif tm_state['current_goal'] == 'b': goal_one_hot[1] = 1.0
        elif tm_state['current_goal'] == 'c': goal_one_hot[2] = 1.0
        else: goal_one_hot[3] = 1.0 # None

        # TM counts (normalized by target_n or max_n)
        target_n = tm_state.get('target_n', 1) # Get target n, default 1 if missing
        norm_a_count = tm_state['a_count'] / target_n if target_n > 0 else 0
        norm_b_count = tm_state['b_count'] / target_n if target_n > 0 else 0
        norm_c_count = tm_state['c_count'] / target_n if target_n > 0 else 0

        # TM phase completion (binary)
        phase_comp_feat = np.array([
            float(tm_state['phase_complete']['a']),
            float(tm_state['phase_complete']['b']),
            float(tm_state['phase_complete']['c'])
        ])

        tm_features = np.concatenate([
            tm_state_one_hot,          # 6
            tm_phase_one_hot,          # 5
            goal_one_hot,              # 4
            np.array([norm_a_count, norm_b_count, norm_c_count]), # 3
            phase_comp_feat            # 3
        ]) # Total TM: 21 features

        # Combine all features
        features = np.concatenate([basic_features, tm_features]) # Total: 18 + 21 = 39 features
    else:
        # If no TM state, pad with zeros or use only basic features
        # Using only basic features might be simpler
        features = basic_features
        # Pad with zeros to maintain consistent length if network expects it
        # padding = np.zeros(21) # Length of TM features
        # features = np.concatenate([basic_features, padding])

    # Ensure float32 for PyTorch
    return features.astype(np.float32)


# Deep Q-Network with attention and dueling architecture
class AttentionDuelingDQN(nn.Module):
    """
    Deep Q-Network with attention mechanism (implicit via feature weighting)
    and dueling architecture.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]): # Reduced hidden layers
        super(AttentionDuelingDQN, self).__init__()

        # Shared layers (can be used before splitting into value/advantage)
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            # nn.Dropout(0.1), # Dropout can sometimes hurt early training
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            # nn.Dropout(0.1)
        )

        # Dueling architecture: Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, 1)
        )

        # Dueling architecture: Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] // 2, output_dim)
        )

    def forward(self, x):
        # Extract features through shared layers
        features = self.feature_layers(x)

        # Calculate value and advantage streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages for Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return qvals


# Experience replay with prioritization (basic PER implementation)
class EnhancedReplayBuffer:
    """
    Replay buffer with prioritized experience replay (PER).
    Hindsight Experience Replay (HER) is more complex and might be overkill here.
    Focusing on PER.
    """
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha # Prioritization exponent
        self.beta_start = beta_start # Initial importance sampling exponent
        self.beta_frames = beta_frames # Frames to anneal beta to 1.0
        self.frame = 1 # For beta annealing

        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0 # Initial max priority

        # Using namedtuple for clearer experience structure
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
        self.priorities.append(self.max_priority) # Add with current max priority

    def _get_beta(self):
        """Calculate the current beta value for importance sampling."""
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        return beta

    def sample(self):
        """Sample a batch of experiences using prioritized replay."""
        if len(self.buffer) < self.batch_size:
             return None # Not enough samples yet

        # Calculate priorities^alpha
        priorities_alpha = np.array(self.priorities) ** self.alpha
        probs = priorities_alpha / np.sum(priorities_alpha)

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs, replace=False) # Sample without replacement

        # Get experiences
        experiences = [self.buffer[i] for i in indices]

        # Calculate importance sampling weights
        beta = self._get_beta()
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= np.max(weights) # Normalize weights

        # Convert to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        weights = torch.from_numpy(weights).float().unsqueeze(1) # Add dimension for broadcasting

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, errors):
        """Update priorities of sampled transitions based on TD errors."""
        for i, idx in enumerate(indices):
            priority = (abs(errors[i]) + 1e-5) ** self.alpha # Add epsilon to avoid zero priority
            # Ensure index is valid (buffer might have shrunk if maxlen reached)
            if idx < len(self.priorities):
                 self.priorities[idx] = priority
                 self.max_priority = max(self.max_priority, priority) # Update max priority seen

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


# Agent with hierarchical policy and curriculum learning
class HierarchicalDQNAgent:
    """
    DQN agent using the AttentionDuelingDQN network and EnhancedReplayBuffer.
    """
    def __init__(self, state_size, action_size, hidden_dims=[256, 128], # Match network
                 gamma=0.99, lr=1e-4, tau=1e-3, # Adjusted LR
                 buffer_size=50000, batch_size=64, # Reduced buffer size
                 update_every=4, double_dqn=True,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100000): # Added PER params
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.double_dqn = double_dqn

        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Q-Networks
        self.qnetwork_local = AttentionDuelingDQN(state_size, action_size, hidden_dims).to(self.device)
        self.qnetwork_target = AttentionDuelingDQN(state_size, action_size, hidden_dims).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict()) # Initialize target same as local
        self.qnetwork_target.eval() # Target network in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Learning rate scheduler (optional, can help stabilize)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)

        # Replay memory (Prioritized)
        self.memory = EnhancedReplayBuffer(buffer_size, batch_size, per_alpha, per_beta_start, per_beta_frames)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

        # Training metrics
        self.loss_history = []
        self.q_value_history = []


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time."""
        # Save experience in replay memory
        # Ensure data is on CPU before adding to buffer (if using GPU)
        state_cpu = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        next_state_cpu = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else next_state

        self.memory.add(state_cpu, action, reward, next_state_cpu, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size: # Check >= batch_size
            experiences = self.memory.sample()
            if experiences: # Check if sampling was successful
                 self.learn(experiences)


    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy with exploration."""
        # Ensure state is a tensor and on the correct device
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnetwork_local.eval() # Set network to evaluation mode for acting
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train() # Set network back to training mode

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using batch of experience tuples from PER."""
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Move tensors to the configured device (CPU or GPU)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)


        # --- Target Q Calculation (using Double DQN logic) ---
        if self.double_dqn:
            # Get best actions for next states from *local* network
            with torch.no_grad():
                best_actions_next = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            # Get Q values for those actions from *target* network
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions_next)
        else:
             # Standard DQN: Get max Q values from target network
             with torch.no_grad():
                 Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states: R + gamma * Q_target(s', a') * (1 - done)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # --- Expected Q Calculation ---
        # Get expected Q values from local model for the actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # --- Loss Calculation (Prioritized Replay) ---
        # Compute TD errors for priority updates
        td_errors = torch.abs(Q_targets - Q_expected).detach() # Keep on device initially

        # Update priorities in replay buffer (pass errors as numpy array)
        self.memory.update_priorities(indices, td_errors.cpu().numpy().flatten())

        # Compute weighted MSE loss: mean(weights * (Q_expected - Q_targets)^2)
        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()

        # Track metrics
        self.loss_history.append(loss.item())
        self.q_value_history.append(Q_expected.mean().item())

        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional but often helpful)
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # --- Update Target Network ---
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        # Step the learning rate scheduler (if used)
        # if hasattr(self, 'scheduler'):
        #     self.scheduler.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters: target = tau*local + (1-tau)*target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        """Save model parameters and optimizer state."""
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history
            # Add memory state if needed, but can be large
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model parameters and optimizer state."""
        try:
            checkpoint = torch.load(filename, map_location=self.device) # Load to the correct device
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_history = checkpoint.get('loss_history', []) # Use .get for backward compatibility
            self.q_value_history = checkpoint.get('q_value_history', [])
            self.qnetwork_local.to(self.device) # Ensure model is on the correct device
            self.qnetwork_target.to(self.device)
            self.qnetwork_target.eval() # Set target to eval mode
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {filename}. Starting from scratch.")
        except Exception as e:
             print(f"Error loading model from {filename}: {e}. Starting from scratch.")


# Training function with curriculum learning
def train_with_curriculum(env_params={'grid_size': 5, 'max_n': 1}, # Start simpler: n=1
                         num_episodes=3000, # Increase episodes
                         eval_interval=100,
                         checkpoint_interval=500,
                         starting_level=0,
                         early_stopping_threshold=0.95, # Higher threshold
                         max_steps_per_episode=150): # Increase max steps
    """
    Train an agent using curriculum learning.
    """
    print("--- Initializing Training ---")
    # Create environment with curriculum learning
    env = SimplifiedTreasureHuntEnv(
        grid_size=env_params['grid_size'],
        max_n=env_params['max_n'],
        max_steps=max_steps_per_episode, # Use adjusted max steps
        curriculum_level=starting_level,
        fixed_positions=(starting_level <= 1), # Use fixed positions for early levels
        render_mode="rgb_array" # Use rgb_array for training, human for eval/viz
    )

    # Create the hierarchical MDPTM_RM wrapper
    mdptm_rm = HierarchicalMDPTM_RM(env)
    mdptm_rm.curriculum_level = starting_level

    # Get state features shape (run one step to be sure)
    observation, info = mdptm_rm.reset()
    tm_state = mdptm_rm.turing_machine.get_state()
    state = extract_features_with_attention(observation, tm_state, grid_size=env.grid_size)
    state_size = len(state)
    action_size = env.action_space.n

    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Environment Params: {env_params}")
    print(f"Max Steps per Episode: {max_steps_per_episode}")

    # Create hierarchical agent
    agent = HierarchicalDQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_dims=[256, 128], # Match network
        gamma=0.99,
        lr=1e-4, # Learning rate
        buffer_size=50000,
        batch_size=128, # Larger batch size
        update_every=4,
        double_dqn=True,
        per_beta_frames= num_episodes * 50 # Anneal beta over roughly half the training
    )

    # Define epsilon schedule (simpler decay)
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.997 # Slower decay

    # Training metrics
    scores = []
    avg_scores = []
    success_rates = []
    curriculum_levels = []
    episode_lengths = []

    # Exploration parameter
    eps = eps_start

    # Main training loop
    print(f"\n--- Starting Training (Max {num_episodes} episodes) ---")
    start_time = time.time()

    for i_episode in range(1, num_episodes + 1):
        # Reset environment and get initial state
        observation, info = mdptm_rm.reset()
        tm_state = mdptm_rm.turing_machine.get_state()
        state = extract_features_with_attention(observation, tm_state, grid_size=env.grid_size)

        # Reset episode variables
        score = 0
        done = False
        steps = 0

        # Episode loop
        while not done:
            # Select action
            action = agent.act(state, eps)

            # Take action in the wrapped environment
            next_observation, reward, terminated, truncated, info = mdptm_rm.step(action)
            done = terminated or truncated

            # Get next state features
            next_tm_state = mdptm_rm.turing_machine.get_state() # Use the latest TM state
            next_state = extract_features_with_attention(next_observation, next_tm_state, grid_size=env.grid_size)

            # Store experience and learn
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward
            steps += 1

            # Break if steps exceed max (already handled by truncated, but safety)
            if steps >= max_steps_per_episode:
                break

        # Store metrics
        scores.append(score)
        episode_lengths.append(steps)
        curriculum_levels.append(mdptm_rm.curriculum_level) # Log level *at end* of episode

        # Calculate running average score (e.g., over last 100 episodes)
        window_size = min(100, len(scores))
        avg_score = np.mean(scores[-window_size:])
        avg_scores.append(avg_score)

        # Calculate success rate (task_complete in info)
        # Need to track success info per episode
        recent_infos = [] # Need to store info dicts to check success
        # This requires modifying the loop or agent.step to pass info back,
        # or tracking success based on reward threshold (less reliable).
        # Let's use reward threshold as a proxy for now:
        is_success = info.get('task_complete', False) # Check the final info dict
        # A better way: track success bools per episode
        # For simplicity now, assume high score means success (tune threshold)
        success_threshold = mdptm_rm.turing_machine.target_n * (10+2) + 50 - max_steps_per_episode*0.1 # Approx score
        recent_successes = sum(1 for s in scores[-window_size:] if s > 0) # Simple proxy: positive score?
        success_rate = recent_successes / window_size if window_size > 0 else 0.0
        success_rates.append(success_rate)


        # Decay epsilon
        eps = max(eps_end, eps_decay * eps)

        # Print progress
        if i_episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Ep {i_episode:>4}/{num_episodes} | Lvl: {mdptm_rm.curriculum_level} | "
                  f"Score: {score:>6.1f} | Avg: {avg_score:>6.1f} | Steps: {steps:>3} | "
                  f"Succ: {success_rate:.2f} | Eps: {eps:.3f} | Time: {elapsed_time:.1f}s")

        # Save checkpoint
        if checkpoint_interval > 0 and i_episode % checkpoint_interval == 0:
            agent.save(f"checkpoint_ep{i_episode}_lvl{mdptm_rm.curriculum_level}.pth")
            # Save metrics as well
            np.savez(f"training_metrics_ep{i_episode}.npz",
                     scores=scores, avg_scores=avg_scores, success_rates=success_rates,
                     curriculum_levels=curriculum_levels, episode_lengths=episode_lengths)

        # Periodic evaluation (on a copy of the env at current level?)
        if eval_interval > 0 and i_episode % eval_interval == 0:
             print(f"\n--- Evaluating at Episode {i_episode} (Level {mdptm_rm.curriculum_level}) ---")
             # Create a separate eval env to avoid interfering with training state
             eval_env = SimplifiedTreasureHuntEnv(
                  grid_size=env_params['grid_size'], max_n=env_params['max_n'],
                  max_steps=max_steps_per_episode,
                  curriculum_level=mdptm_rm.curriculum_level, # Eval at current level
                  fixed_positions=(mdptm_rm.curriculum_level <= 1),
                  render_mode="rgb_array" # No rendering during eval usually
             )
             eval_wrapper = HierarchicalMDPTM_RM(eval_env)
             eval_success_rate = evaluate_agent(eval_wrapper, agent, num_episodes=20) # Increase eval episodes
             print(f"--- Evaluation Complete: Success Rate = {eval_success_rate:.2f} ---\n")

             # Check for early stopping
             if eval_success_rate >= early_stopping_threshold and mdptm_rm.curriculum_level >= env_params['max_n'] * 2 : # Stop if high success at harder levels
                 print(f"\nEarly stopping criterion met at episode {i_episode}!")
                 break

    # End of training
    print("\n--- Training Finished ---")
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # Save final model
    agent.save("final_model.pth")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final_env = SimplifiedTreasureHuntEnv(
         grid_size=env_params['grid_size'], max_n=env_params['max_n'],
         max_steps=max_steps_per_episode,
         curriculum_level=mdptm_rm.curriculum_level, # Eval at final level
         fixed_positions=(mdptm_rm.curriculum_level <= 1),
         render_mode="rgb_array"
    )
    final_wrapper = HierarchicalMDPTM_RM(final_env)
    final_success_rate = evaluate_agent(final_wrapper, agent, num_episodes=50) # More eval episodes
    print(f"--- Final Evaluation Complete: Success Rate = {final_success_rate:.2f} ---")

    # Plot training metrics
    plot_training_metrics(scores, avg_scores, success_rates, curriculum_levels, episode_lengths)

    return agent, scores, success_rates, curriculum_levels


# Evaluation function
def evaluate_agent(wrapper, agent, num_episodes=10, render=False, render_mode='human'):
    """Evaluate agent performance over multiple episodes."""
    successes = 0
    total_steps = 0

    # Temporarily set render mode if needed
    original_render_mode = wrapper.env.render_mode
    if render:
        wrapper.env.render_mode = render_mode

    for i in range(num_episodes):
        # Reset environment using the wrapper
        observation, info = wrapper.reset()
        tm_state = wrapper.turing_machine.get_state() # Get initial TM state
        state = extract_features_with_attention(observation, tm_state, grid_size=wrapper.env.grid_size)

        # Episode variables
        done = False
        episode_steps = 0
        episode_success = False

        # Episode loop
        while not done:
            if render:
                 wrapper.render() # Use wrapper's render
                 time.sleep(0.2) # Slow down rendering

            # Select action (greedy policy, eps=0)
            action = agent.act(state, eps=0.0)

            # Take action using the wrapper
            next_observation, reward, terminated, truncated, info = wrapper.step(action)
            done = terminated or truncated

            # Get next state features
            next_tm_state = wrapper.turing_machine.get_state()
            next_state = extract_features_with_attention(next_observation, next_tm_state, grid_size=wrapper.env.grid_size)

            # Update state
            state = next_state
            episode_steps += 1

            # Check if task was completed successfully using info dict
            if info.get('task_complete', False):
                 episode_success = True
                 # No need to break here, let episode finish naturally or truncate

            # Safety break if steps exceed max (should be handled by truncated)
            if episode_steps >= wrapper.env.max_steps * 1.1: # Allow some buffer
                print("Warning: Evaluation episode exceeded max steps unexpectedly.")
                break


        if episode_success:
            successes += 1
        total_steps += episode_steps

        # Print episode result if verbose or rendering
        if render or num_episodes <= 10:
            result = "Success!" if episode_success else "Failed"
            reason = ""
            if not episode_success:
                 reason = f" ({info.get('error', 'unknown reason')})" if info else ""

            print(f"Eval Ep {i+1}: {result}{reason} ({episode_steps} steps)")

    # Restore original render mode
    if render:
        wrapper.env.render_mode = original_render_mode

    # Calculate success rate
    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    avg_steps = total_steps / num_episodes if num_episodes > 0 else 0

    print(f"Evaluation Summary: {successes}/{num_episodes} successful ({success_rate:.2f}). Avg steps: {avg_steps:.1f}")

    return success_rate


# Visualization functions
def plot_training_metrics(scores, avg_scores, success_rates, curriculum_levels, episode_lengths):
    """Plot training metrics over time."""
    print("\nGenerating training plots...")
    episodes = range(1, len(scores) + 1)

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, axs = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

    # Plot scores
    axs[0].plot(episodes, scores, alpha=0.4, label='Episode Score', color='lightblue')
    axs[0].plot(episodes, avg_scores, linewidth=2, label='Avg Score (100 ep)', color='blue')
    axs[0].set_ylabel('Score')
    axs[0].set_title('Training Scores')
    axs[0].legend()
    axs[0].grid(True)

    # Plot success rate
    axs[1].plot(episodes, success_rates, label='Success Rate (100 ep window)', color='green')
    axs[1].set_ylabel('Success Rate')
    axs[1].set_title('Approximate Success Rate During Training')
    axs[1].set_ylim([-0.05, 1.05])
    axs[1].grid(True)

    # Plot episode lengths
    axs[2].plot(episodes, episode_lengths, label='Episode Length', color='orange', alpha=0.5)
    # Add rolling average for length
    if len(episode_lengths) >= 100:
         avg_lengths = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
         axs[2].plot(range(100, len(episode_lengths) + 1), avg_lengths, color='red', label='Avg Length (100 ep)')
    axs[2].set_ylabel('Steps')
    axs[2].set_title('Episode Lengths')
    axs[2].legend()
    axs[2].grid(True)


    # Plot curriculum level
    axs[3].plot(episodes, curriculum_levels, label='Curriculum Level', color='purple', marker='.', linestyle='')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Level')
    axs[3].set_title('Curriculum Level Progression')
    axs[3].grid(True)
    # Make y-axis integer ticks for levels
    max_level = max(curriculum_levels) if curriculum_levels else 0
    axs[3].set_yticks(np.arange(0, max_level + 2))


    plt.tight_layout()
    try:
        plt.savefig('training_metrics.png')
        print("Training metrics plot saved to 'training_metrics.png'")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Optionally display plot immediately
    plt.close() # Close plot to free memory


def visualize_episode(wrapper, agent, max_steps=150):
    """Run and visualize one episode using the wrapper and agent."""
    print("\n--- Visualizing Agent Behavior ---")
    # Reset environment using the wrapper
    observation, info = wrapper.reset()
    tm_state = wrapper.turing_machine.get_state()
    state = extract_features_with_attention(observation, tm_state, grid_size=wrapper.env.grid_size)

    # Episode variables
    done = False
    episode_steps = 0
    total_reward = 0
    episode_success = False

    # Set render mode to human for visualization
    original_render_mode = wrapper.env.render_mode
    wrapper.env.render_mode = 'human'

    # Episode loop
    while not done and episode_steps < max_steps:
        # Render the current state
        wrapper.render()
        time.sleep(0.3) # Adjust speed as needed

        # Select action (greedy policy)
        action = agent.act(state, eps=0.0)
        action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
        print(f"Step: {episode_steps+1}, Action: {action_names[action]}")

        # Take action
        next_observation, reward, terminated, truncated, info = wrapper.step(action)
        done = terminated or truncated
        total_reward += reward

        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        print(f"TM State: {info.get('tm_state_val')}, Phase: {info.get('tm_phase')}, Goal: {info.get('tm_goal')}")
        print(f"Counts: a={info.get('tm_a')}, b={info.get('tm_b')}, c={info.get('tm_c')} (n={info.get('n_value')})")
        print("-" * 30)


        # Get next state features
        next_tm_state = wrapper.turing_machine.get_state()
        next_state = extract_features_with_attention(next_observation, next_tm_state, grid_size=wrapper.env.grid_size)

        # Update state
        state = next_state
        episode_steps += 1

        # Check if task completed
        if info.get('task_complete', False):
            episode_success = True

    # Final frame render
    wrapper.render()

    # Restore original render mode
    wrapper.env.render_mode = original_render_mode

    # Print episode summary
    print("\n--- Episode End ---")
    print(f"Finished in {episode_steps} steps.")
    print(f"Total reward: {total_reward:.2f}")
    result = "Success!" if episode_success else "Failure"
    reason = ""
    if not episode_success and info:
         reason = f" ({info.get('error', 'unknown reason')})"
    print(f"Result: {result}{reason}")

    return episode_success

# Run main experiment
def run_main_experiment():
    """Run the main experiment with our enhanced approach."""
    # Set parameters
    env_params = {
        'grid_size': 6,  # Slightly larger grid
        'max_n': 2,      # Max n = 2 (a^2 b^2 c^2 is quite hard)
    }
    training_episodes = 5000 # More episodes needed for n=2
    max_steps_episode = 200 # More steps allowed

    # Train with curriculum learning
    print("=" * 60)
    print(" STARTING TRAINING ".center(60, "="))
    print("=" * 60)
    agent, scores, success_rates, curriculum_levels = train_with_curriculum(
        env_params=env_params,
        num_episodes=training_episodes,
        eval_interval=250, # Evaluate less frequently
        checkpoint_interval=1000, # Checkpoint less frequently
        starting_level=0,
        max_steps_per_episode=max_steps_episode,
        early_stopping_threshold=0.90 # Adjust threshold
    )

    # Create environment and wrapper for final visualization/evaluation
    final_level = max(curriculum_levels) if curriculum_levels else 0
    print(f"\n--- Preparing for Final Demonstration (Level {final_level}) ---")
    demo_env = SimplifiedTreasureHuntEnv(
        grid_size=env_params['grid_size'],
        max_n=env_params['max_n'],
        max_steps=max_steps_episode,
        curriculum_level=final_level,
        fixed_positions=(final_level <= 1),
        render_mode="human" # Set for visualization
    )
    demo_wrapper = HierarchicalMDPTM_RM(demo_env)

    # Load the best model (or final model)
    # agent.load("final_model.pth") # Or load a specific checkpoint if better

    # Demonstrate agent behavior
    print("\n--- Demonstrating Agent Behavior (Final Model) ---")
    for i in range(3):  # Show 3 demonstration episodes
        print(f"\n--- DEMONSTRATION {i+1} ---")
        success = visualize_episode(demo_wrapper, agent, max_steps=max_steps_episode)
        if not success and i < 2:
            print("Attempting another demonstration...")
        elif i == 2:
             print("End of demonstrations.")

    demo_wrapper.close() # Close the demo environment

    return {
        'agent': agent,
        'scores': scores,
        'success_rates': success_rates,
        'curriculum_levels': curriculum_levels,
        'final_success_rate': success_rates[-1] if success_rates else 0, # Approx final rate
        'max_level_reached': final_level
    }


# Entry point
if __name__ == "__main__":
    print("+" * 60)
    print(" Enhanced a^n b^n c^n Treasure Hunt Experiment ".center(60, "+"))
    print("+" * 60)

    results = run_main_experiment()

    print("\n" + "=" * 60)
    print(" EXPERIMENT SUMMARY ".center(60, "="))
    print("=" * 60)
    final_rate = results['final_success_rate']
    max_level = results['max_level_reached']
    print(f"Approximate Final Success Rate (last 100 ep): {final_rate:.2f}")
    print(f"Maximum Curriculum Level Reached: {max_level}")
    print(f"Total Training Episodes: {len(results['scores'])}")
    print("=" * 60)

    if final_rate >= 0.8 and max_level >= 2: # Adjust success criteria as needed
        print("\nEXPERIMENT LIKELY SUCCESSFUL: Agent shows good learning progress!")
    elif final_rate > 0.5:
        print("\nEXPERIMENT PARTIALLY SUCCESSFUL: Agent learned but may need more tuning/training.")
    else:
        print("\nEXPERIMENT NEEDS REVIEW: Agent struggled to learn effectively.")

    print("\nExperiment finished.")