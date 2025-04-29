import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(DuelingDQN, self).__init__()
        
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
        """Forward pass through the network."""
        features = self.feature_layers(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))
    
    def update(self, states, actions, rewards, next_states, dones, optimizer, device, gamma=0.99):
        """Update the network parameters using a batch of experiences."""
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Get current Q values
        current_q_values = self(states).gather(1, actions)
        
        # Compute target Q values (Double DQN)
        with torch.no_grad():
            # Select actions using the current network
            next_actions = self(next_states).argmax(dim=1, keepdim=True)
            # Evaluate Q values using current network
            next_q_values = self(next_states).gather(1, next_actions)
            # Compute target Q values
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to help stabilize training
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling experiences.
    """
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self):
        """Sample a batch of experiences from the buffer."""
        experiences = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
        
        # Unzip the experiences
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences])
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
