import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from collections import OrderedDict

class MetaPolicy(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        # Extract config parameters
        custom_config = model_config.get("custom_model_config", {})
        self.initial_weights = custom_config.get("initial_weights", [])
        self.num_bs = custom_config.get("num_bs", 4)
        self.num_ue = custom_config.get("num_ue", 20)
        
        # Calculate input size - one UE's observation size
        if isinstance(obs_space, gym.spaces.Dict):
            # For Dict space, we need the size of a single agent's observation
            # This should be 2*num_bs+1
            # input_size = 4 * self.num_bs + 4
            input_size = 2*self.num_bs + 2
        else:
            input_size = np.prod(obs_space.shape)
            
        # print(f"Calculated input size: {input_size}")
        
        # Enhanced network architecture for better performance
        hidden_size = 64  # Add a hidden layer for more expressive policy
        
        # Policy network for actions - each UE chooses from num_bs actions
        self.policy_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_bs+1)  # Output should match number of BS options
        )
        
        # Value network (critic) - also with a hidden layer
        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Store current value function output
        self._cur_value = None
        
        # Initialize weights using metaheuristic solution if available
        if self.initial_weights:
            self._apply_initial_weights()
            
        # Print network shapes for debugging
        # print(f"Policy network: {self.policy_network}")
        # print(f"Value network: {self.value_network}")
        
    def _apply_initial_weights(self):
        """Apply initial weights to bias the policy"""
        # Each UE should have its own policy preferences
        with torch.no_grad():
            # Get the last layer of the policy network
            policy_output_layer = self.policy_network[-1]
            
            # Initialize with small random weights for exploration
            policy_output_layer.weight.data.normal_(0.0, 0.01)
            policy_output_layer.bias.data.fill_(0.0)
            
            # If we have initial weights from metaheuristic
            if isinstance(self.initial_weights, list) and len(self.initial_weights) > 0:
                # Determine which UE this policy is for based on available context
                # In MARL with parameter sharing, we can't know for sure,
                # so we use a more general approach
                
                # Count the frequency of each BS in the solution
                bs_counts = np.zeros(self.num_bs)
                for bs_idx in self.initial_weights:
                    if 0 <= bs_idx < self.num_bs:
                        bs_counts[bs_idx] += 1
                
                # Bias toward less congested BSs
                total_ues = sum(bs_counts)
                if total_ues > 0:
                    for bs_idx in range(self.num_bs):
                        # Lower allocation ratio = higher bias
                        congestion_factor = 1.0 - (bs_counts[bs_idx] / total_ues)
                        policy_output_layer.bias.data[bs_idx] = congestion_factor * 1.0 # Stronger bias toward less congested BSs
                
                # print(f"Applied metaheuristic bias based on BS congestion")
                
    def forward(self, input_dict, state, seq_lens):
        # Get observation from input dict
        obs = input_dict["obs"]
        
        # Debug: Check observation shape and values
        # print(f"Forward input shape: {obs.shape if hasattr(obs, 'shape') else 'dict'}")
        # if isinstance(obs, torch.Tensor) and obs.numel() > 0:
            # print(f"Forward input stats: min={obs.min().item():.4f}, max={obs.max().item():.4f}, "
            #     f"mean={obs.mean().item():.4f}, has_nan={torch.isnan(obs).any().item()}")
        
        # Handle different input types
        if isinstance(obs, dict) or isinstance(obs, OrderedDict):
            # Debug: Print dict keys and their shapes
            # print(f"Forward dict keys: {list(obs.keys())}")
            for k, v in obs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k} shape: {v.shape}")
                    
            # In MARL, each agent should only receive its own observation
            # Convert all values to tensors and flatten if needed
            x = torch.cat([torch.tensor(v).flatten() for v in obs.values()])
        else:
            # Already a tensor
            x = obs.float() if isinstance(obs, torch.Tensor) else torch.FloatTensor(obs)
            
        # Ensure batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Debug: Check processed input tensor
        # print(f"Processed input shape: {x.shape}")
                
        # Forward passes
        logits = self.policy_network(x)
        self._cur_value = self.value_network(x).squeeze(-1)
        
        # # Debug: Check outputs
        # print(f"Logits shape: {logits.shape}, values shape: {self._cur_value.shape}")
        # print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
        #     f"mean={logits.mean().item():.4f}, has_nan={torch.isnan(logits).any().item()}")
        
        return logits, state

    def value_function(self):
        """Return value function output for current observation"""
        # This is required for PPO training
        assert self._cur_value is not None, "value function not calculated"
        
        # Debug: Check value function output
        # if isinstance(self._cur_value, torch.Tensor) and self._cur_value.numel() > 0:
        #     print(f"Value function stats: min={self._cur_value.min().item():.4f}, "
        #         f"max={self._cur_value.max().item():.4f}, mean={self._cur_value.mean().item():.4f}, "
        #         f"has_nan={torch.isnan(self._cur_value).any().item()}")
        
        return self._cur_value 