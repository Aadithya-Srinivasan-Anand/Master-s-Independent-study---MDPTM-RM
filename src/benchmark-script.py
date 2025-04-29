import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from collections import deque
import gymnasium as gym

# Import environment and model implementations
from parenthesis_balancing_env import ParenthesisBalancingEnv, HierarchicalMDPTM_RM_Parenthesis
from neural_turing_machine import NeuralTuringMachine, NTMAgent

# Import feature extraction and evaluation utilities
from feature_extraction import extract_features_with_attention
from evaluation import evaluate_model, demonstrate_model


def run_benchmark(env_params={'grid_size': 7, 'max_depth': 3}, 
                  num_episodes=1000, 
                  eval_interval=100,
                  curriculum_levels=[0, 1, 2, 3]):
    """
    Run comprehensive benchmark comparing MDPTM-RM against Neural Turing Machine
    on the Parenthesis Balancing task.
    """
    print("\n====== MDPTM-RM vs Neural Turing Machine Benchmark ======\n")
    
    # Print experiment parameters
    print(f"Environment: Parenthesis Balancing")
    print(f"Grid Size: {env_params['grid_size']}x{env_params['grid_size']}")
    print(f"Max Nesting Depth: {env_params['max_depth']}")
    print(f"Training Episodes: {num_episodes}")
    print(f"Curriculum Levels: {curriculum_levels}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize results dictionary
    results = {
        'mdptm_rm': {'success_rates': [], 'learning_speed': None, 'eval_by_level': {}},
        'ntm': {'success_rates': [], 'learning_speed': None, 'eval_by_level': {}}
    }
    
    # ===== PART 1: MDPTM-RM Training and Evaluation =====
    print("\n----- Training MDPTM-RM -----")
    
    # Create environment with MDPTM-RM wrapper
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
    
    # Setup MDPTM-RM model and training
    observation, _ = mdptm_env.reset()
    pda_state = mdptm_env.pda.get_state()
    state = extract_features_with_attention(observation, pda_state, grid_size=mdptm_env.env.grid_size)
    state_size = len(state)
    action_size = mdptm_env.action_space.n
    
    # Create DQN model for MDPTM-RM
    from dqn_model import DuelingDQN, ReplayBuffer
    
    mdptm_model = DuelingDQN(state_size, action_size).to(device)
    mdptm_optimizer = optim.Adam(mdptm_model.parameters(), lr=1e-4)
    mdptm_buffer = ReplayBuffer(50000, 64)
    
    # Training loop
    mdptm_start_time = time.time()
    mdptm_success_history = []
    epsilon = 1.0
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        observation, _ = mdptm_env.reset()
        pda_state = mdptm_env.pda.get_state()
        state = extract_features_with_attention(observation, pda_state, grid_size=mdptm_env.env.grid_size)
        
        # Episode variables
        episode_reward = 0
        done = False
        episode_success = False
        
        # Episode loop
        while not done:
            # Select action (epsilon-greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            if random.random() > epsilon:
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
            
            # Store transition
            mdptm_buffer.add(state, action, reward, next_state, float(done))
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Check for success
            if info.get('balanced', False):
                episode_success = True
            
            # Train model if enough samples
            if len(mdptm_buffer) >= 64:
                batch = mdptm_buffer.sample()
                mdptm_model.update(*batch, optimizer=mdptm_optimizer, device=device)
        
        # Record success
        mdptm_success_history.append(1 if episode_success else 0)
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
        
        # Print progress
        if episode % 10 == 0:
            success_rate = np.mean(mdptm_success_history[-100:]) if len(mdptm_success_history) >= 100 else np.mean(mdptm_success_history)
            print(f"Episode {episode}/{num_episodes} | Success Rate: {success_rate:.2f} | Epsilon: {epsilon:.3f}")
            
            # Check if we've reached success threshold
            if success_rate >= 0.5 and results['mdptm_rm']['learning_speed'] is None:
                results['mdptm_rm']['learning_speed'] = episode
                print(f"MDPTM-RM reached 50% success rate at episode {episode}")
    
    # Record training time
    mdptm_training_time = time.time() - mdptm_start_time
    print(f"\nMDPTM-RM training completed in {mdptm_training_time:.2f} seconds")
    
    # Store success rates
    results['mdptm_rm']['success_rates'] = mdptm_success_history
    
    # ===== PART 2: Neural Turing Machine Training and Evaluation =====
    print("\n----- Training Neural Turing Machine -----")
    
    # Create environment for NTM
    ntm_env = ParenthesisBalancingEnv(
        grid_size=env_params['grid_size'],
        max_depth=env_params['max_depth'],
        max_steps=200,
        curriculum_level=0,
        fixed_positions=True,
        render_mode="rgb_array"
    )
    
    # Setup NTM model and training
    observation, _ = ntm_env.reset()
    state = extract_features_with_attention(observation, grid_size=ntm_env.grid_size)
    state_size = len(state)
    action_size = ntm_env.action_space.n
    
    # Create NTM agent
    ntm_agent = NTMAgent(
        input_size=state_size,
        output_size=action_size,
        device=device,
        memory_n=64,
        memory_m=16,
        learning_rate=5e-4
    )
    
    # Training loop
    ntm_start_time = time.time()
    ntm_success_history = []
    epsilon = 1.0
    
    for episode in range(1, num_episodes + 1):
        # Reset environment
        observation, _ = ntm_env.reset()
        state = extract_features_with_attention(observation, grid_size=ntm_env.grid_size)
        
        # Reset NTM
        ntm_agent.reset()
        
        # Episode variables
        episode_reward = 0
        done = False
        episode_success = False
        
        # Storage for episode transitions
        episode_transitions = []
        
        # Episode loop
        while not done:
            # Select action (epsilon-greedy)
            action = ntm_agent.select_action(state, epsilon)
            
            # Take action
            next_observation, reward, terminated, truncated, info = ntm_env.step(action)
            next_state = extract_features_with_attention(next_observation, grid_size=ntm_env.grid_size)
            done = terminated or truncated
            
            # Store transition
            episode_transitions.append((state, action, reward, next_state, float(done)))
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Check for success
            if info.get('balanced', False):
                episode_success = True
        
        # Add episode transitions to NTM buffer
        for transition in episode_transitions:
            ntm_agent.store_transition(*transition)
        
        # Train NTM
        if len(ntm_agent.memory_buffer) >= 64:
            ntm_agent.learn(batch_size=64)
        
        # Record success
        ntm_success_history.append(1 if episode_success else 0)
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
        
        # Print progress
        if episode % 10 == 0:
            success_rate = np.mean(ntm_success_history[-100:]) if len(ntm_success_history) >= 100 else np.mean(ntm_success_history)
            print(f"Episode {episode}/{num_episodes} | Success Rate: {success_rate:.2f} | Epsilon: {epsilon:.3f}")
            
            # Check if we've reached success threshold
            if success_rate >= 0.5 and results['ntm']['learning_speed'] is None:
                results['ntm']['learning_speed'] = episode
                print(f"NTM reached 50% success rate at episode {episode}")
    
    # Record training time
    ntm_training_time = time.time() - ntm_start_time
    print(f"\nNTM training completed in {ntm_training_time:.2f} seconds")
    
    # Store success rates
    results['ntm']['success_rates'] = ntm_success_history
    
    # ===== PART 3: Evaluation Across Curriculum Levels =====
    print("\n----- Evaluating Models Across Curriculum Levels -----")
    
    for level in curriculum_levels:
        print(f"\nEvaluating at curriculum level {level}:")
        
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
        results['mdptm_rm']['eval_by_level'][level] = mdptm_success_rate
        
        # Evaluate NTM
        print("Evaluating NTM...")
        ntm_success_rate = evaluate_model(ntm_agent.ntm, 'ntm', ntm_eval_env, device, num_episodes=20)
        results['ntm']['eval_by_level'][level] = ntm_success_rate
        
        # Print comparative results
        print(f"Level {level} Results - MDPTM-RM: {mdptm_success_rate:.2f}, NTM: {ntm_success_rate:.2f}")
    
    # ===== PART 4: Analysis and Visualization =====
    print("\n----- Benchmark Analysis -----")
    
    # Learning speed comparison
    if results['mdptm_rm']['learning_speed'] and results['ntm']['learning_speed']:
        mdptm_speed = results['mdptm_rm']['learning_speed'] 
        ntm_speed = results['ntm']['learning_speed']
        
        speed_ratio = mdptm_speed / ntm_speed if mdptm_speed > ntm_speed else ntm_speed / mdptm_speed
        faster_model = "NTM" if mdptm_speed > ntm_speed else "MDPTM-RM"
        
        print(f"\nLearning Speed Comparison:")
        print(f"MDPTM-RM reached 50% success at episode: {mdptm_speed}")
        print(f"NTM reached 50% success at episode: {ntm_speed}")
        print(f"{faster_model} learned {speed_ratio:.2f}x faster")
    else:
        print("\nOne or both models did not reach 50% success threshold")
    
    # Final success rate comparison
    mdptm_final = np.mean(results['mdptm_rm']['success_rates'][-100:])
    ntm_final = np.mean(results['ntm']['success_rates'][-100:])
    
    print(f"\nFinal Success Rates (last 100 episodes):")
    print(f"MDPTM-RM: {mdptm_final:.2f}")
    print(f"NTM: {ntm_final:.2f}")
    
    # Curriculum level performance
    print("\nPerformance by Curriculum Level:")
    for level in curriculum_levels:
        mdptm_level = results['mdptm_rm']['eval_by_level'][level]
        ntm_level = results['ntm']['eval_by_level'][level]
        print(f"Level {level}: MDPTM-RM = {mdptm_level:.2f}, NTM = {ntm_level:.2f}, Diff = {mdptm_level - ntm_level:+.2f}")
    
    # Training time comparison
    print(f"\nTraining Time:")
    print(f"MDPTM-RM: {mdptm_training_time:.2f}s")
    print(f"NTM: {ntm_training_time:.2f}s")
    
    # ===== PART 5: Visualization =====
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    
    # Calculate moving averages
    window = 100
    mdptm_ma = [np.mean(results['mdptm_rm']['success_rates'][max(0,i-window):i+1]) 
                for i in range(len(results['mdptm_rm']['success_rates']))]
    ntm_ma = [np.mean(results['ntm']['success_rates'][max(0,i-window):i+1]) 
              for i in range(len(results['ntm']['success_rates']))]
    
    plt.plot(range(1, len(mdptm_ma)+1), mdptm_ma, 'b-', label='MDPTM-RM')
    plt.plot(range(1, len(ntm_ma)+1), ntm_ma, 'r-', label='NTM')
    
    plt.title('Success Rate Comparison')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (moving avg)')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves_comparison.png')
    
    # Plot curriculum level performance
    plt.figure(figsize=(10, 6))
    
    levels = sorted(results['mdptm_rm']['eval_by_level'].keys())
    mdptm_levels = [results['mdptm_rm']['eval_by_level'][l] for l in levels]
    ntm_levels = [results['ntm']['eval_by_level'][l] for l in levels]
    
    x = np.arange(len(levels))
    width = 0.35
    
    plt.bar(x - width/2, mdptm_levels, width, label='MDPTM-RM', color='blue')
    plt.bar(x + width/2, ntm_levels, width, label='NTM', color='red')
    
    plt.title('Performance by Curriculum Level')
    plt.xlabel('Curriculum Level')
    plt.ylabel('Success Rate')
    plt.xticks(x, [str(l) for l in levels])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('curriculum_level_comparison.png')
    
    # ===== PART 6: Demonstration =====
    print("\n----- Model Demonstration -----")
    
    # Create demo environments
    mdptm_demo_env = HierarchicalMDPTM_RM_Parenthesis(
        ParenthesisBalancingEnv(
            grid_size=env_params['grid_size'],
            max_depth=env_params['max_depth'],
            max_steps=200,
            curriculum_level=3,  # Higher difficulty for demo
            fixed_positions=False,
            render_mode="human"
        )
    )
    
    ntm_demo_env = ParenthesisBalancingEnv(
        grid_size=env_params['grid_size'],
        max_depth=env_params['max_depth'],
        max_steps=200,
        curriculum_level=3,  # Same difficulty
        fixed_positions=False,
        render_mode="human"
    )
    
    # Run demonstration
    print("\nDemonstrating MDPTM-RM...")
    mdptm_success = demonstrate_model(mdptm_model, 'mdptm_rm', mdptm_demo_env, device)
    
    print("\nDemonstrating NTM...")
    ntm_success = demonstrate_model(ntm_agent.ntm, 'ntm', ntm_demo_env, device)
    
    # Summary
    print("\n====== Benchmark Summary ======")
    print(f"Learning Speed: {'MDPTM-RM' if mdptm_speed < ntm_speed else 'NTM'} was faster")
    print(f"Final Performance: {'MDPTM-RM' if mdptm_final > ntm_final else 'NTM'} achieved higher success rate")
    print(f"Curriculum Generalization: {'MDPTM-RM' if np.mean(mdptm_levels) > np.mean(ntm_levels) else 'NTM'} showed better generalization")
    print(f"Demo Results: MDPTM-RM {'succeeded' if mdptm_success else 'failed'}, NTM {'succeeded' if ntm_success else 'failed'}")
    
    overall_winner = "MDPTM-RM" if (mdptm_speed < ntm_speed and mdptm_final > ntm_final) or np.mean(mdptm_levels) > np.mean(ntm_levels) else "NTM"
    print(f"\nOverall Winner: {overall_winner}")
    
    return results


if __name__ == "__main__":
    # Run the benchmark
    results = run_benchmark(
        env_params={'grid_size': 7, 'max_depth': 3},
        num_episodes=1000,
        eval_interval=100,
        curriculum_levels=[0, 1, 2, 3]
    )
    
    print("\nBenchmark completed. Results saved to learning_curves_comparison.png and curriculum_level_comparison.png")
