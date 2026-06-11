import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

from torch.utils.data import Dataset
import numpy as np
import torch

class BCDataset(Dataset):
    def __init__(self, bc_list, num_ue):
        """
        bc_list: output of env.build_bc_dataset()
        num_ue: number of UEs in the environment
        """
        self.samples = []
        for obs_dict, action_dict in bc_list:
            for ue_idx in range(num_ue):
                state_vec  = obs_dict[f"ue_{ue_idx}"]           # e.g. numpy array shape = [obs_dim]
                action_int = action_dict[f"ue_{ue_idx}"]        # int in [0..num_bs-1]
                self.samples.append((state_vec.astype(np.float32), action_int))
        self.num_ue = num_ue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action = self.samples[idx]
        # Return as (FloatTensor, LongTensor) for cross‐entropy
        return torch.from_numpy(state), torch.tensor(action, dtype=torch.long)    
