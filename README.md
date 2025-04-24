A comparative study of the Markov Decision Process with Turing Machine and Reward Machine (MDPTM-RM) approach versus Neural Turing Machine (NTM) on context-free language recognition tasks.

## Overview

This project implements and compares two approaches to solving formal language recognition tasks:

1. **MDPTM-RM**: A hierarchical approach that combines:
   - Pushdown Automaton (PDA) for explicit context-free language recognition
   - Reward Machine for structured reward shaping
   - Reinforcement Learning for adaptive behavior

2. **Neural Turing Machine (NTM)**: An end-to-end differentiable architecture with:
   - External memory with soft read/write operations
   - LSTM controller network
   - Content and location-based addressing mechanisms

We demonstrate that for context-free language tasks, explicitly incorporating formal language theory into model design (MDPTM-RM) dramatically outperforms general neural architectures (NTM) in terms of learning efficiency, final performance, and generalization.

## Task: Parenthesis Balancing

The primary experimental task is parenthesis balancing, a classic context-free language recognition problem:

- Agent navigates a grid environment to collect opening '(' and closing ')' parentheses
- Must collect them in a valid balanced order (e.g., "()()()","((()))","(()())")
- Task difficulty (curriculum level) progressively increases:
  - Level 0-1: Simple patterns with fixed positions
  - Level 2-3: More complex patterns with semi-random positions
  - Level 4+: Complex nested patterns with fully random positions

## Key Results

![Success Rate Comparison](learning_curves_comparison.png)

The comparison shows MDPTM-RM (blue) significantly outperforms NTM (red) in:
- **Learning efficiency**: Reaches high success rate much faster
- **Final performance**: Achieves ~80% success vs NTM's near-zero performance
- **Stability**: Maintains consistent performance throughout training

![Training Metrics](training_metrics.png)

MDPTM-RM's training metrics show effective curriculum learning:
- Steady improvement in score and success rate
- Successful progression through increasing difficulty levels
- Adaptation to more complex tasks requiring longer episodes

## Repository Structure

```
├── parenthesis-balancing-env.py    # Main grid environment implementation
├── evaluation.py                   # Model evaluation and demonstration utilities
├── feature_extraction_file.py      # Feature engineering with attention mechanisms
├── dqn_model.py                    # DQN model implementation for MDPTM-RM
├── neural-turing-machine-implementation.py  # NTM implementation
├── comparative-analysis.py         # Analysis and visualization functions
├── benchmark-script.py             # Scripts for running comparative experiments
└── MDPTM-RM.py                     # Core implementation of the MDPTM-RM approach
```

## Installation and Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/mdptm-rm-vs-ntm.git
cd mdptm-rm-vs-ntm

# Install dependencies
pip install torch numpy matplotlib gymnasium
```

## Running Experiments

To run the benchmark comparison:

```python
from benchmark_script import run_benchmark

results = run_benchmark(
    env_params={'grid_size': 7, 'max_depth': 3},
    num_episodes=1000,
    eval_interval=100,
    curriculum_levels=[0, 1, 2, 3]
)
```

This will train both MDPTM-RM and NTM models on the parenthesis balancing task and generate the comparison plots.

## Implementation Details

### Environment

The `ParenthesisBalancingEnv` class implements a grid-world environment where an agent must navigate to collect opening and closing parentheses in a valid balanced order. It includes:

- Grid state management with configurable size
- Agent movement and item collection logic
- Curriculum levels with varying complexity
- Basic reward mechanisms

### MDPTM-RM Framework

The framework consists of three key components:

1. **Pushdown Automaton (PDA)**:
   ```python
   class ParenthesisBalancingPDA:
       def process_symbol(self, symbol):
           if symbol == '(':
               # Push to stack for opening parenthesis
               self.stack.append('(')
               self.max_stack_depth = max(self.max_stack_depth, len(self.stack))
           elif symbol == ')':
               # Pop for closing parenthesis if stack not empty
               if len(self.stack) > 0:
                   self.stack.pop()
               else:
                   # Unmatched closing parenthesis - reject
                   self.current_state = 'qreject'
   ```

2. **Reward Machine**:
   ```python
   class ParenthesisRewardMachine:
       def get_reward(self, old_pda_state, new_pda_state, env_reward, distances=None, info=None):
           final_reward = env_reward
           
           # Add distance-based shaping if distances provided
           if distances is not None and current_goal in distances:
               current_distance = distances[current_goal]
               prev_distance = self.last_min_distances[current_goal]
               
               # Reward for getting closer to target
               if current_distance < prev_distance:
                   improvement = prev_distance - current_distance
                   distance_reward = self.distance_improvement_factor * improvement
                   final_reward += distance_reward
   ```

3. **Hierarchical Wrapper**:
   ```python
   class HierarchicalMDPTM_RM_Parenthesis:
       def step(self, action):
           # Take action in environment
           observation, env_reward, terminated, truncated, info = self.env.step(action)
           
           # Check if item was collected
           collected_symbol = info.get('collected')
           if collected_symbol is not None:
               # Update PDA with the collected symbol
               self.pda.process_symbol(collected_symbol)
           
           # Calculate shaped reward
           shaped_reward = self.reward_machine.get_reward(
               self.last_tm_state,
               current_tm_state,
               env_reward,
               min_distances_after,
               info
           )
           
           return observation, shaped_reward, terminated, truncated, full_info
   ```

### Feature Extraction with Attention

```python
def extract_features_with_attention(observation, pda_state=None, grid_size=None):
    # Determine current goal based on PDA state
    current_goal = None
    if pda_state is not None:
        # If stack is empty, we need an opening
        if pda_state['stack_depth'] == 0:
            current_goal = 'opening'
        # If stack is not empty, we might want a closing
        else:
            current_goal = 'closing'
    
    # Set attention weights: higher for current goal
    attention_weights = {'opening': 0.3, 'closing': 0.3}
    if current_goal in attention_weights:
        attention_weights[current_goal] = 1.0
        
    # Calculate direction vectors weighted by attention
    # and other features...
```

### Neural Models

1. **DuelingDQN** (for MDPTM-RM):
   ```python
   class DuelingDQN(nn.Module):
       def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
           # Feature extraction layers
           self.feature_layers = nn.Sequential(
               nn.Linear(input_dim, hidden_dims[0]),
               nn.ReLU(),
               nn.Linear(hidden_dims[0], hidden_dims[1]),
               nn.ReLU()
           )
           
           # Value stream
           self.value_stream = nn.Sequential(
               nn.Linear(hidden_dims[1], 32),
               nn.ReLU(),
               nn.Linear(32, 1)
           )
           
           # Advantage stream
           self.advantage_stream = nn.Sequential(
               nn.Linear(hidden_dims[1], 32),
               nn.ReLU(),
               nn.Linear(32, output_dim)
           )
       
       def forward(self, x):
           features = self.feature_layers(x)
           
           value = self.value_stream(features)
           advantages = self.advantage_stream(features)
           
           # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
           return value + (advantages - advantages.mean(dim=1, keepdim=True))
   ```

2. **Neural Turing Machine**:
   ```python
   class NeuralTuringMachine(nn.Module):
       def forward(self, x):
           # Controller processes input
           controller_out, self.controller_state = self.controller(x_augmented)
           
           # Memory addressing and reading
           read_content_weights = self._content_addressing(read_key, read_beta)
           self.read_weights = self._sharpen(read_shifted, read_gamma)
           read_vector = torch.matmul(self.read_weights, self.memory)
           
           # Memory writing
           erase = torch.outer(self.write_weights, erase_vector)
           add = torch.outer(self.write_weights, add_vector)
           self.memory = self.memory * (1 - erase) + add
   ```

## Theoretical Analysis

The performance gap can be explained by fundamental differences in how these models approach the context-free language recognition task:

| MDPTM-RM | Neural Turing Machine |
|----------|------------------------|
| Explicit stack operations | Must learn memory management from scratch |
| Direct alignment with formal language theory | General architecture lacks domain-specific structure |
| Hierarchical decision-making | End-to-end learning with high parameter complexity |
| Domain-specific reward shaping | Generic reward signals |
| Interpretable internal state | Black-box computation |

MDPTM-RM's advantage comes from explicitly incorporating the PDA formalism, which is theoretically proven to be the correct computational model for context-free languages.

## Running the Benchmark

To run the main comparison benchmark and generate the plots shown above:

```python
# Import and run the benchmark
from benchmark_script import run_benchmark

# Run the benchmark with default parameters
results = run_benchmark()

# Or customize parameters
results = run_benchmark(
    env_params={'grid_size': 7, 'max_depth': 3},
    num_episodes=1000,
    eval_interval=100,
    curriculum_levels=[0, 1, 2, 3]
)
```

This will train both models and generate the comparative learning curves and curriculum level comparison plots.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
