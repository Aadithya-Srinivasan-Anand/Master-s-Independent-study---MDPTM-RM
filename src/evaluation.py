import numpy as np
import torch
import time
from feature_extraction import extract_features_with_attention

def evaluate_model(model, model_name, env, device, num_episodes=10):
    """
    Evaluate a model's performance on the parenthesis balancing task.
    
    Args:
        model: The model to evaluate (either MDPTM-RM or NTM)
        model_name: String identifier for the model type
        env: Environment to evaluate in
        device: Torch device
        num_episodes: Number of episodes to evaluate
        
    Returns:
        success_rate: Fraction of successful episodes
    """
    successes = 0
    
    # Ensure model is in evaluation mode
    if hasattr(model, 'eval'):
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
        
        # Reset NTM if applicable
        if model_name == 'ntm' and hasattr(model, 'reset'):
            model.reset()
        
        # Episode variables
        done = False
        episode_success = False
        
        # Episode loop
        while not done:
            # Select action (greedy policy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
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
            
            # Check if task completed successfully
            if info.get('balanced', False):
                episode_success = True
        
        if episode_success:
            successes += 1
    
    # Set model back to training mode if applicable
    if hasattr(model, 'train'):
        model.train()
    
    # Calculate success rate
    success_rate = successes / num_episodes
    
    return success_rate


def demonstrate_model(model, model_name, env, device, max_steps=200):
    """
    Run a visual demonstration of the model on the parenthesis balancing task.
    
    Args:
        model: The model to demonstrate
        model_name: String identifier for the model type
        env: Environment for demonstration
        device: Torch device
        max_steps: Maximum steps per episode
        
    Returns:
        success: Boolean indicating if the demonstration was successful
    """
    print(f"\n--- Demonstrating {model_name} ---")
    
    # Ensure model is in evaluation mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Reset environment
    if model_name == 'mdptm_rm':
        observation, info = env.reset()
        pda_state = env.pda.get_state()
        state = extract_features_with_attention(observation, pda_state, grid_size=env.env.grid_size)
    else:
        observation, info = env.reset()
        state = extract_features_with_attention(observation, grid_size=env.grid_size)
    
    # Reset NTM if applicable
    if model_name == 'ntm' and hasattr(model, 'reset'):
        model.reset()
    
    # Episode variables
    steps = 0
    total_reward = 0
    done = False
    success = False
    
    # Display initial state
    env.render()
    print(f"Target pattern: {info.get('target_pattern', '')}")
    time.sleep(1)
    
    # Episode loop
    while not done and steps < max_steps:
        # Select action (greedy policy)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_values = model(state_tensor)
        action = torch.argmax(action_values).item()
        
        # Display action
        action_names = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
        print(f"Step {steps+1}: Action = {action_names[action]}")
        
        # Take action
        if model_name == 'mdptm_rm':
            next_observation, reward, terminated, truncated, info = env.step(action)
            pda_state = env.pda.get_state()
            next_state = extract_features_with_attention(next_observation, pda_state, grid_size=env.env.grid_size)
            
            # Show PDA state
            print(f"PDA State: {pda_state['state']}, Stack: {''.join(pda_state['stack'])}")
        else:
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = extract_features_with_attention(next_observation, grid_size=env.grid_size)
        
        done = terminated or truncated
        
        # Update state and tracking variables
        state = next_state
        steps += 1
        total_reward += reward
        
        # Check for success
        if info.get('balanced', False):
            success = True
            print("\nSUCCESS! Pattern correctly balanced.")
        
        # Show render and pause
        env.render()
        time.sleep(0.5)
    
    # Final summary
    print("\n--- Demonstration Summary ---")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Result: {'Success' if success else 'Failure'}")
    
    if not success and 'error' in info:
        print(f"Error: {info['error']}")
    
    # Set model back to training mode if applicable
    if hasattr(model, 'train'):
        model.train()
    
    return success
