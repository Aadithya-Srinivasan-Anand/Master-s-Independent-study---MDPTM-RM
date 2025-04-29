import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralTuringMachine(nn.Module):
    """
    A more complete implementation of Neural Turing Machine (NTM) based on the original paper.
    This version includes more sophisticated memory addressing and controller mechanisms.
    """
    def __init__(self, input_size, output_size, 
                 memory_n=128, memory_m=20,  # n = # of memory locations, m = vector size
                 controller_hidden_size=100,
                 controller_layers=1):
        super(NeuralTuringMachine, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.memory_n = memory_n  # Number of memory locations
        self.memory_m = memory_m  # Size of each memory vector
        
        # Initialize memory
        self.memory = torch.zeros(memory_n, memory_m)
        
        # Initialize read weights and write weights
        self.read_weights = torch.zeros(memory_n)
        self.write_weights = torch.zeros(memory_n)
        
        # Previous read vector
        self.last_read = torch.zeros(memory_m)
        
        # Controller (LSTM)
        self.controller = nn.LSTM(
            input_size=input_size + memory_m,  # Input + previous read
            hidden_size=controller_hidden_size,
            num_layers=controller_layers,
            batch_first=True
        )
        
        self.controller_hidden_size = controller_hidden_size
        self.controller_state = None  # (h, c) for LSTM
        
        # Memory addressing parameters
        # Parameters for content-based addressing
        self.key_fc = nn.Linear(controller_hidden_size, memory_m)
        self.beta_fc = nn.Linear(controller_hidden_size, 1)  # Key strength
        
        # Parameters for location-based addressing
        self.gate_fc = nn.Linear(controller_hidden_size, 1)  # Interpolation gate
        self.shift_fc = nn.Linear(controller_hidden_size, 3)  # Shift weighting [-1, 0, 1]
        self.gamma_fc = nn.Linear(controller_hidden_size, 1)  # Sharpening
        
        # Parameters for memory writing
        self.erase_fc = nn.Linear(controller_hidden_size, memory_m)  # Erase vector
        self.add_fc = nn.Linear(controller_hidden_size, memory_m)  # Add vector
        
        # Output layer
        self.output_fc = nn.Linear(controller_hidden_size + memory_m, output_size)
        
        # For device tracking
        self.device = torch.device("cpu")
    
    def reset(self):
        """Reset the memory and controller state"""
        self.memory = torch.zeros(self.memory_n, self.memory_m).to(self.device)
        self.read_weights = torch.zeros(self.memory_n).to(self.device)
        self.write_weights = torch.zeros(self.memory_n).to(self.device)
        self.last_read = torch.zeros(self.memory_m).to(self.device)
        self.controller_state = None
        
        # Initialize weights to focus on the first memory location
        self.read_weights[0] = 1.0
        self.write_weights[0] = 1.0
    
    def _content_addressing(self, key, beta):
        """
        Content-based addressing: Focus on memory locations similar to the key
        key: shape (memory_m)
        beta: scalar key strength
        """
        # Cosine similarity between key and each memory location
        similarity = F.cosine_similarity(
            key.unsqueeze(0),     # shape (1, memory_m)
            self.memory,          # shape (memory_n, memory_m)
            dim=1
        )
        
        # Apply key strength (beta) and softmax
        weights = F.softmax(beta * similarity, dim=0)
        return weights
    
    def _interpolate(self, content_weights, prev_weights, g):
        """
        Interpolate between content-based and previous weights
        g: scalar interpolation gate (0-1)
        """
        return g * content_weights + (1 - g) * prev_weights
    
    def _shift(self, weights, shift_weights):
        """
        Circular convolution for location-based addressing
        shift_weights: shape (3) for shifts {-1, 0, 1}
        """
        n = self.memory_n
        shifted_weights = torch.zeros(n).to(self.device)
        
        # Apply each shift
        for i in range(3):
            shift = i - 1  # {-1, 0, 1}
            for j in range(n):
                shifted_weights[(j + shift) % n] += weights[j] * shift_weights[i]
                
        return shifted_weights
    
    def _sharpen(self, weights, gamma):
        """
        Sharpen the weight distribution
        gamma: scalar >= 1
        """
        w_pow = weights.pow(gamma)
        return w_pow / w_pow.sum()
    
    def forward(self, x):
        """
        Forward pass through the NTM
        x: input shape (batch_size, input_size) - but we only support batch_size=1 for simplicity
        """
        # Ensure x is 3D with batch_size=1, sequence_length=1
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and time dims
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dim
        
        batch_size = x.size(0)
        
        # Concatenate input with previous read
        x_augmented = torch.cat([x, self.last_read.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)], dim=2)
        
        # Run controller
        controller_out, self.controller_state = self.controller(x_augmented, self.controller_state)
        controller_out = controller_out.squeeze(1)  # Remove time dimension
        
        # Memory addressing - Read
        read_key = torch.tanh(self.key_fc(controller_out))
        read_beta = F.softplus(self.beta_fc(controller_out))
        read_gate = torch.sigmoid(self.gate_fc(controller_out))
        read_shift = F.softmax(self.shift_fc(controller_out), dim=1)
        read_gamma = 1 + F.softplus(self.gamma_fc(controller_out))
        
        # Read from memory (using addressing mechanisms)
        read_content_weights = self._content_addressing(read_key.squeeze(0), read_beta.squeeze(0))
        read_interpolated = self._interpolate(read_content_weights, self.read_weights, read_gate.squeeze(0))
        read_shifted = self._shift(read_interpolated, read_shift.squeeze(0))
        self.read_weights = self._sharpen(read_shifted, read_gamma.squeeze(0))
        
        # Read from memory
        read_vector = torch.matmul(self.read_weights, self.memory)
        self.last_read = read_vector.clone()
        
        # Memory addressing - Write
        write_key = torch.tanh(self.key_fc(controller_out))
        write_beta = F.softplus(self.beta_fc(controller_out))
        write_gate = torch.sigmoid(self.gate_fc(controller_out))
        write_shift = F.softmax(self.shift_fc(controller_out), dim=1)
        write_gamma = 1 + F.softplus(self.gamma_fc(controller_out))
        
        # Write to memory (using addressing mechanisms)
        write_content_weights = self._content_addressing(write_key.squeeze(0), write_beta.squeeze(0))
        write_interpolated = self._interpolate(write_content_weights, self.write_weights, write_gate.squeeze(0))
        write_shifted = self._shift(write_interpolated, write_shift.squeeze(0))
        self.write_weights = self._sharpen(write_shifted, write_gamma.squeeze(0))
        
        # Get erase and add vectors
        erase_vector = torch.sigmoid(self.erase_fc(controller_out)).squeeze(0)
        add_vector = torch.tanh(self.add_fc(controller_out)).squeeze(0)
        
        # Erase and add to memory
        erase = torch.outer(self.write_weights, erase_vector)
        add = torch.outer(self.write_weights, add_vector)
        
        self.memory = self.memory * (1 - erase) + add
        
        # Produce output
        output = torch.cat([controller_out, read_vector.unsqueeze(0)], dim=1)
        output = self.output_fc(output)
        
        return output.squeeze(0)  # Remove batch dimension
    
    def to(self, device):
        """Move the model to the specified device"""
        self.device = device
        if hasattr(self, 'memory'):
            self.memory = self.memory.to(device)
        if hasattr(self, 'read_weights'):
            self.read_weights = self.read_weights.to(device)
        if hasattr(self, 'write_weights'):
            self.write_weights = self.write_weights.to(device)
        if hasattr(self, 'last_read'):
            self.last_read = self.last_read.to(device)
        return super(NeuralTuringMachine, self).to(device)


class NTMAgent:
    """
    Agent that uses a Neural Turing Machine for decision making.
    """
    def __init__(self, input_size, output_size, device='cpu',
                 memory_n=128, memory_m=20,
                 learning_rate=0.0001):
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        # Create the NTM model
        self.ntm = NeuralTuringMachine(
            input_size=input_size,
            output_size=output_size,
            memory_n=memory_n,
            memory_m=memory_m
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.ntm.parameters(), lr=learning_rate)
        
        # For training
        self.memory_buffer = []
        
    def select_action(self, state, epsilon=0.0):
        """Select an action using epsilon-greedy policy"""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            with torch.no_grad():
                action_values = self.ntm(state_tensor)
            return torch.argmax(action_values).item()
        else:
            return np.random.randint(self.output_size)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer"""
        self.memory_buffer.append((state, action, reward, next_state, done))
        
    def learn(self, batch_size=32, gamma=0.99):
        """Train the network using experiences from the buffer"""
        if len(self.memory_buffer) < batch_size:
            return
        
        # Sample batch
        if len(self.memory_buffer) > batch_size:
            batch = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
            experiences = [self.memory_buffer[i] for i in batch]
        else:
            experiences = self.memory_buffer
        
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones, dtype=np.uint8)).unsqueeze(1).to(self.device)
        
        # Reset NTM state for each batch element (simplified)
        self.ntm.reset()
        
        # Get current Q values
        current_q_values = []
        for state in states:
            q_value = self.ntm(state).gather(0, actions[0])
            current_q_values.append(q_value)
        current_q_values = torch.stack(current_q_values)
        
        # Reset NTM again for next state calculations
        self.ntm.reset()
        
        # Get next Q values
        next_q_values = []
        with torch.no_grad():
            for next_state in next_states:
                q_value = self.ntm(next_state).max().unsqueeze(0)
                next_q_values.append(q_value)
        next_q_values = torch.stack(next_q_values)
        
        # Compute target Q values
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.ntm.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def reset(self):
        """Reset the NTM's memory and state"""
        self.ntm.reset()


# Helper functions for comparative training with NTM

def train_ntm_agent(env, ntm_agent, num_episodes=500, epsilon_start=1.0, 
                    epsilon_end=0.01, epsilon_decay=0.995, 
                    batch_size=32, gamma=0.99):
    """Train the NTM agent on the parenthesis balancing environment"""
    # Metrics
    scores = []
    avg_scores = []
    success_rates = []
    epsilon = epsilon_start
    
    for i_episode in range(1, num_episodes + 1):
        # Reset environment
        observation, info = env.reset()
        state = extract_features_with_attention(observation, grid_size=env.grid_size)
        
        # Reset NTM
        ntm_agent.reset()
        
        # Episode variables
        score = 0
        done = False
        episode_success = False
        
        # Experience buffer for this episode
        episode_buffer = []
        
        # Episode loop
        while not done:
            # Select action
            action = ntm_agent.select_action(state, epsilon)
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = extract_features_with_attention(next_observation, grid_size=env.grid_size)
            done = terminated or truncated
            
            # Store transition
            episode_buffer.append((state, action, reward, next_state, done))
            
            # Update state and score
            state = next_state
            score += reward
            
            # Check if task completed successfully
            if info.get('balanced', False):
                episode_success = True
        
        # Add experience to agent's buffer
        ntm_agent.memory_buffer.extend(episode_buffer)
        
        # Train NTM
        if len(ntm_agent.memory_buffer) >= batch_size:
            ntm_agent.learn(batch_size, gamma)
            
            # Limit buffer size to prevent memory issues
            if len(ntm_agent.memory_buffer) > 10000:
                ntm_agent.memory_buffer = ntm_agent.memory_buffer[-10000:]
        
        # Store metrics
        scores.append(score)
        
        # Calculate running average score
        window_size = min(100, len(scores))
        avg_score = np.mean(scores[-window_size:])
        avg_scores.append(avg_score)
        
        # Approximate success rate
        success_rates.append(1.0 if episode_success else 0.0)
        recent_success_rate = np.mean(success_rates[-window_size:])
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        
        # Print progress
        if i_episode % 10 == 0:
            print(f"NTM Episode {i_episode}/{num_episodes} | Score: {score:.1f} | Avg: {avg_score:.1f} | Success: {recent_success_rate:.2f} | Epsilon: {epsilon:.3f}")
    
    return {
        'scores': scores,
        'avg_scores': avg_scores,
        'success_rates': success_rates,
        'final_success_rate': np.mean(success_rates[-100:]) if len(success_rates) >= 100 else np.mean(success_rates)
    }


def evaluate_ntm_agent(env, ntm_agent, num_episodes=10):
    """Evaluate NTM agent performance"""
    successes = 0
    scores = []
    
    for i in range(num_episodes):
        # Reset environment
        observation, info = env.reset()
        state = extract_features_with_attention(observation, grid_size=env.grid_size)
        
        # Reset NTM
        ntm_agent.reset()
        
        # Episode variables
        score = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action (greedy policy)
            action = ntm_agent.select_action(state, epsilon=0.0)
            
            # Take action
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = extract_features_with_attention(next_observation, grid_size=env.grid_size)
            done = terminated or truncated
            
            # Update state and score
            state = next_state
            score += reward
            
            # Check if task completed successfully
            if info.get('balanced', False):
                successes += 1
                break
        
        scores.append(score)
    
    # Calculate success rate and average score
    success_rate = successes / num_episodes
    avg_score = np.mean(scores)
    
    return success_rate, avg_score


def comparative_ntm_experiment(env_params={'grid_size': 7, 'max_depth': 3},
                              num_episodes=2000, batch_size=32):
    """
    Run a focused experiment comparing MDPTM-RM against NTM on the parenthesis balancing task.
    This highlights the differences in how these models approach context-free languages.
    """
    print("\n--- Starting Comparative NTM Experiment ---")
    
    # Create environments
    mdptm_env = HierarchicalMDPTM_RM_Parenthesis(
        ParenthesisBalancingEnv(
            grid_size=env_params['grid_size'],
            max_depth=env_params['max_depth'],
            max_steps=200,
            curriculum_level=0,
            fixed_positions=True,
            render_mode="rgb_array"
        )
    )
    
    ntm_env = ParenthesisBalancingEnv(
            grid_size=env_params['grid_size'],
            max_depth=env_params['max_depth'],
            max_steps=200,
            curriculum_level=0,
            fixed_positions=True,
            render_mode="rgb_array"
    )
    
    # Get observation dimension
    observation, _ = ntm_env.reset()
    state = extract_features_with_attention(observation, grid_size=ntm_env.grid_size)
    state_size = len(state)
    action_size = ntm_env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create MDPTM-RM model
    mdptm_observation, _ = mdptm_env.reset()
    pda_state = mdptm_env.pda.get_state()
    mdptm_state = extract_features_with_attention(mdptm_observation, pda_state, grid_size=mdptm_env.env.grid_size)
    mdptm_state_size = len(mdptm_state)
    
    mdptm_model = AttentionDuelingDQN(mdptm_state_size, action_size, hidden_dims=[128, 64]).to(device)
    mdptm_optimizer = torch.optim.Adam(mdptm_model.parameters(), lr=1e-4)
    mdptm_memory = EnhancedReplayBuffer(50000, batch_size, alpha=0.6, beta_start=0.4)
    
    # Create NTM agent
    ntm_agent = NTMAgent(
        input_size=state_size,
        output_size=action_size,
        device=device,
        memory_n=64,    # Memory size parameters
        memory_m=16,    # Can be tuned
        learning_rate=5e-4
    )
    
    # Training parameters
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    gamma = 0.99
    tau = 1e-3
    
    # Metrics tracking
    mdptm_metrics = {
        'scores': [],
        'avg_scores': [],
        'success_rates': []
    }
    
    ntm_metrics = {
        'scores': [],
        'avg_scores': [],
        'success_rates': []
    }
    
    # Training loops
    print("\n--- Training MDPTM-RM Model ---")
    
    # MDPTM-RM training
    mdptm_eps = eps_start
    for i_episode in range(1, num_episodes + 1):
        # Reset environment
        observation, info = mdptm_env.reset()
        pda_state = mdptm_env.pda.get_state()
        state = extract_features_with_attention(observation, pda_state, grid_size=mdptm_env.env.grid_size)
        
        # Episode variables
        score = 0
        done = False
        success = False
        
        # Experience buffer for this episode
        experiences = []
        
        # Episode loop
        while not done:
            # Select action (epsilon-greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            if random.random() > mdptm_eps:
                with torch.no_grad():
                    action_values = mdptm_model(state_tensor)
                action = torch.argmax(action_values).item()
            else:
                action = random.randrange(action_size)
            
            # Take action
            next_observation, reward, terminated, truncated, info = mdptm_env.step(action)
            next_pda_state = mdptm_env.pda.get_state()
            next_state = extract_features_with_attention(next_observation, next_pda_state, grid_size=mdptm_env.env.grid_size)
            done = terminated or truncated
            
            # Store experience
            experiences.append((state, action, reward, next_state, float(done)))
            
            # Update state and score
            state = next_state
            score += reward
            
            # Check for success
            if info.get('balanced', False):
                success = True
        
        # Add experiences to memory buffer
        for exp in experiences:
            mdptm_memory.add(*exp)
        
        # Train the model if enough samples
        if len(mdptm_memory) >= batch_size:
            # Sample from memory
            experiences = mdptm_memory.sample()
            if experiences:
                states, actions, rewards, next_states, dones, indices, weights = experiences
                
                # Move to device
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                weights = weights.to(device)
                
                # Get Q values
                q_values = mdptm_model(states).gather(1, actions)
                
                # Get next Q values (Double DQN)
                with torch.no_grad():
                    next_actions = mdptm_model(next_states).argmax(dim=1, keepdim=True)
                    next_q_values = mdptm_model(next_states).gather(1, next_actions)
                    targets = rewards + (gamma * next_q_values * (1 - dones))
                
                # Calculate loss and update priorities
                td_errors = torch.abs(targets - q_values).detach()
                mdptm_memory.update_priorities(indices, td_errors.cpu().numpy().flatten())
                
                # Weighted MSE loss
                loss = (weights * F.mse_loss(q_values, targets, reduction='none')).mean()
                
                # Optimize
                mdptm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdptm_model.parameters(), 1.0)
                mdptm_optimizer.step()
        
        # Store metrics
        mdptm_metrics['scores'].append(score)
        
        # Calculate running average score
        window_size = min(100, len(mdptm_metrics['scores']))
        avg_score = np.mean(mdptm_metrics['scores'][-window_size:])
        mdptm_metrics['avg_scores'].append(avg_score)
        
        # Store success
        mdptm_metrics['success_rates'].append(1.0 if success else 0.0)
        recent_success_rate = np.mean(mdptm_metrics['success_rates'][-window_size:])
        
        # Decay epsilon
        mdptm_eps = max(eps_end, eps_decay * mdptm_eps)
        
        # Print progress
        if i_episode % 10 == 0:
            print(f"MDPTM Episode {i_episode}/{num_episodes} | Score: {score:.1f} | Avg: {avg_score:.1f} | Success: {recent_success_rate:.2f} | Epsilon: {mdptm_eps:.3f}")
    
    # Train the NTM agent
    print("\n--- Training NTM Agent ---")
    ntm_metrics = train_ntm_agent(
        ntm_env, 
        ntm_agent, 
        num_episodes=num_episodes,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=eps_decay,
        batch_size=batch_size,
        gamma=gamma
    )
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    
    # Create evaluation environments with multiple difficulties
    eval_levels = [0, 1, 2, 3]  # Different curriculum levels to test
    
    mdptm_eval_results = []
    ntm_eval_results = []
    
    for level in eval_levels:
        print(f"\nEvaluating at curriculum level {level}")
        
        # Create evaluation environments
        mdptm_eval_env = HierarchicalMDPTM_RM_Parenthesis(
            ParenthesisBalancingEnv(
                grid_size=env_params['grid_size'],
                max_depth=env_params['max_depth'],
                max_steps=200,
                curriculum_level=level,
                fixed_positions=(level <= 1),
                render_mode="rgb_array"
            )
        )
        
        ntm_eval_env = ParenthesisBalancingEnv(
            grid_size=env_params['grid_size'],
            max_depth=env_params['max_depth'],
            max_steps=200,
            curriculum_level=level,
            fixed_positions=(level <= 1),
            render_mode="rgb_array"
        )
        
        # Evaluate MDPTM-RM
        print("Evaluating MDPTM-RM...")
        mdptm_success_rate = evaluate_model(mdptm_model, 'mdptm_rm', mdptm_eval_env, device, num_episodes=20)
        mdptm_eval_results.append((level, mdptm_success_rate))
        print(f"MDPTM-RM Success Rate: {mdptm_success_rate:.2f}")
        
        # Evaluate NTM
        print("Evaluating NTM...")
        ntm_success_rate, ntm_avg_score = evaluate_ntm_agent(ntm_eval_env, ntm_agent, num_episodes=20)
        ntm_eval_results.append((level, ntm_success_rate))
        print(f"NTM Success Rate: {ntm_success_rate:.2f}, Avg Score: {ntm_avg_score:.1f}")
    
    # Analyze and report results
    print("\n=== Comparative Results Analysis ===")
    print("\nSuccess Rates by Curriculum Level:")
    print("Level | MDPTM-RM | NTM")
    print("-" * 30)
    
    for i, level in enumerate(eval_levels):
        mdptm_rate = mdptm_eval_results[i][1]
        ntm_rate = ntm_eval_results[i][1]
        print(f"{level:5d} | {mdptm_rate:.2f}    | {ntm_rate:.2f}")
    
    # Calculate overall performance
    mdptm_overall = np.mean([r[1] for r in mdptm_eval_results])
    ntm_overall = np.mean([r[1] for r in ntm_eval_results])
    
    print(f"\nOverall Success Rate: MDPTM-RM = {mdptm_overall:.2f}, NTM = {ntm_overall:.2f}")
    
    # Learning speed comparison
    mdptm_learning_speed = next((i for i, r in enumerate(mdptm_metrics['success_rates']) if r > 0.5), num_episodes)
    ntm_learning_speed = next((i for i, r in enumerate(ntm_metrics['success_rates']) if r > 0.5), num_episodes)
    
    print(f"\nEpisodes to reach 0.5 success rate:")
    print(f"MDPTM-RM: {mdptm_learning_speed}")
    print(f"NTM: {ntm_learning_speed}")
    
    # Plot comparative learning curves
    plot_comparative_curves(mdptm_metrics, ntm_metrics)
    
    # Return collected data for further analysis
    return {
        'mdptm_metrics': mdptm_metrics,
        'ntm_metrics': ntm_metrics,
        'mdptm_eval': mdptm_eval_results,
        'ntm_eval': ntm_eval_results,
        'mdptm_model': mdptm_model,
        'ntm_agent': ntm_agent
    }


def plot_comparative_curves(mdptm_metrics, ntm_metrics):
    """Plot comparative learning curves for MDPTM-RM and NTM"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Prepare data
    mdptm_len = len(mdptm_metrics['avg_scores'])
    ntm_len = len(ntm_metrics['avg_scores'])
    
    episodes_mdptm = list(range(1, mdptm_len + 1))
    episodes_ntm = list(range(1, ntm_len + 1))
    
    # Success rate moving average
    window = 100
    mdptm_success_ma = [np.mean(mdptm_metrics['success_rates'][max(0, i-window):i+1]) 
                        for i in range(len(mdptm_metrics['success_rates']))]
    ntm_success_ma = [np.mean(ntm_metrics['success_rates'][max(0, i-window):i+1]) 
                        for i in range(len(ntm_metrics['success_rates']))]
    
    # Plot success rates
    plt.figure(figsize=(12, 6))
    plt.plot(episodes_mdptm, mdptm_success_ma, label='MDPTM-RM', color='blue', linewidth=2)
    plt.plot(episodes_ntm, ntm_success_ma, label='NTM', color='red', linewidth=2)
    plt.title('Success Rate Comparison: MDPTM-RM vs Neural Turing Machine')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (moving avg)')
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('success_comparison.png')
    plt.close()
    
    # Plot average scores
    plt.figure(figsize=(12, 6))
    plt.plot(episodes_mdptm, mdptm_metrics['avg_scores'], label='MDPTM-RM', color='blue', linewidth=2)
    plt.plot(episodes_ntm, ntm_metrics['avg_scores'], label='NTM', color='red', linewidth=2)
    plt.title('Average Score Comparison: MDPTM-RM vs Neural Turing Machine')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('score_comparison.png')
    plt.close()
    
    print("\nComparison plots saved as 'success_comparison.png' and 'score_comparison.png'")
