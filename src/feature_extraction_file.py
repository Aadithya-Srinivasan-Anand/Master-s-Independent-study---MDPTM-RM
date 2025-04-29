import numpy as np

def extract_features_with_attention(observation, pda_state=None, grid_size=None):
    """
    Extract features with attention mechanisms for the parenthesis balancing task.
    
    Args:
        observation: Environment observation dictionary
        pda_state: Optional PDA state information (for MDPTM-RM)
        grid_size: Optional grid size if not inferrable from observation
        
    Returns:
        features: Numpy array of extracted features
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
        stack_status                      # max_depth (variable)
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
