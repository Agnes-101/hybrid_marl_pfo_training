import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

from typing import Dict
import numpy as np

class PolicyMappingManager:
    def __init__(self, bs_positions: np.ndarray, initial_ue_positions: Dict[str, np.ndarray]):
        """
        Initialize policy mapping manager
        
        Args:
            bs_positions: Array of shape (num_bs, 2) with BS positions
            initial_ue_positions: Dict mapping agent_id to initial position

        
        # Maps UEs to their closest base station based on Euclidean distance (spatial proximity).
            The positions of all UEs are tracked and each UE assigned to the nearest
        base station using Euclidean (straight-line) distance between UE coordinates
        and BS coordinates. 
        
            As UEs move over time, their associations are continuously
        updated to reflect their current closest base station           

        The resulting mapping is used to determine which base station policy
        (e.g., scheduling or resource allocation strategy) is applied to each UE in
        a multi-cell wireless environment, enabling location-aware policy selection.


        """
        self.bs_positions = bs_positions
        self.ue_positions = initial_ue_positions.copy()
        
    def update_ue_positions(self, new_positions: Dict[str, np.ndarray]):
        """Update UE positions when they move"""
        self.ue_positions.update(new_positions)
        
    def get_closest_bs(self, agent_id: str) -> int:
        """
        Find closest BS to a UE based on current tracked positions
        
        Args:
            agent_id: UE identifier (e.g., "ue0")
            
        Returns:
            BS index (0 = macro, 1-3 = small cells)
        """
        # Wherever you're setting up ue_positions
        # Extract the numeric part from "ue_0" -> 0
               
            
        if agent_id not in self.ue_positions:
            print(f"Warning: {agent_id} position not found, defaulting to macro")
            return 0
            
        ue_pos = self.ue_positions[agent_id]
        
        # Calculate distances to all BSs
        distances = [
            np.linalg.norm(ue_pos - bs_pos) 
            for bs_pos in self.bs_positions
        ]
        
        return int(np.argmin(distances))
    
    def get_policy_distribution(self) -> Dict[str, int]:
        """Get count of UEs assigned to each policy"""

        distribution = {"bs_0_policy": 0, "bs_1_policy": 0, "bs_2_policy": 0, "bs_3_policy": 0}
        
        for agent_id in self.ue_positions.keys():
            closest_bs = self.get_closest_bs(agent_id)
            policy_name = f"bs_{closest_bs}_policy"
            distribution[policy_name] += 1
            
        return distribution
    
    def log_policy_assignments(self):
        """Log current policy assignments for debugging"""
        assignments = {}
        for agent_id in self.ue_positions.keys():
            closest_bs = self.get_closest_bs(agent_id)
            assignments[agent_id] = f"bs_{closest_bs}_policy"
        
        distribution = self.get_policy_distribution()
        # print(f"Policy distribution: {distribution}")
        return assignments
                    