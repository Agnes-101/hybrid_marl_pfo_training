import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
# print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.bc.bc_dataset import BCDataset


def build_bc_dataset(env, num_samples: int,
                     out_path: str = "bc_partial.pkl"):
    """
        Build a BC dataset and periodically flush to disk so that if the process
        crashes halfway, you still have the samples collected so far.
    """
    dataset = []
    # If there’s a partial file, load it and resume
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            dataset = pickle.load(f)
        start_idx = len(dataset)
        print(f"[BC] Resuming from {start_idx} / {num_samples} samples.")
    else:
        start_idx = 0
    for i in range(start_idx, num_samples):
        # 1) Reset env for a new random state
        seed_for_this_episode = 42 + i
        obs_dict, _ = env.bc_reset(seed=seed_for_this_episode)

        # 2) Run your metaheuristic and get initial_solution
        meta_out = env._execute_metaheuristic_phase(env.config["metaheuristic"])
        initial_solution = meta_out["solution"]

        # 3) Grab new observations
        obs_dict = env._get_obs()

            # 4) Convert to action_dict
        action_dict = {
                f"ue_{ue_idx}": int(bs_idx)
                for ue_idx, bs_idx in enumerate(initial_solution)
            }

            # 5) Append and immediately pickle the growing list
        dataset.append((obs_dict, action_dict))
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"[BC] Collected {i+1}/{num_samples} samples, saved to {out_path}")

        return dataset
    
def run_bc_pretraining(env,
               trainer,
               num_bc_samples=500,
               bc_epochs=5,
               batch_size=128):
    """
    1) Build a BC dataset by running the metaheuristic on random states.
    2) BC‐train the shared policy head inside the RLlib MetaPolicy.
    3) Save a checkpoint of the pretrained policy network, return its path.
    """
    num_ue = env.num_ue

        # Step A: build the raw (obs_dict, action_dict) dataset
    bc_list = build_bc_dataset(env, num_samples=num_bc_samples)
    bc_dataset = BCDataset(bc_list, num_ue=num_ue)
    bc_loader  = DataLoader(bc_dataset, batch_size=batch_size, shuffle=True)

    # Step B: grab the MetaPolicy model from RLlib’s trainer
    # rl_policy   = trainer.get_policy()   # RLlib Policy object
    rl_policy = trainer.get_policy("default_policy")
    if rl_policy is None:
        raise RuntimeError("BC failed: cannot find policy 'shared_policy'")
    torch_model = rl_policy.model        # instance of MetaPolicy

        # Freeze all parameters except the policy head
    for name, param in torch_model.named_parameters():
        if "policy_network" not in name:
            param.requires_grad = False

    # Build an optimizer just for the policy_network parameters
    optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, torch_model.parameters()),
            lr=3e-4
        )

        # Step C: supervised BC loop (cross-entropy)
    for epoch in range(bc_epochs):
        total_loss = 0.0
        for state_batch, action_batch in bc_loader:
            # state_batch: [B, obs_dim]; action_batch: [B]
            logits = torch_model.policy_network(state_batch)  # [B, num_bs]
            loss_bc = F.cross_entropy(logits, action_batch)
            optimizer.zero_grad()
            loss_bc.backward()
            optimizer.step()
            total_loss += loss_bc.item()
        avg_loss = total_loss / len(bc_loader)
        print(f"[BC] Epoch {epoch+1}/{bc_epochs}, avg CE loss = {avg_loss:.4f}")

    # Step D: unfreeze all parameters (so PPO can later update both actor & critic)
    for param in torch_model.parameters():
        param.requires_grad = True

    # Step E: save a checkpoint of the pretrained policy network
    checkpoint_path = trainer.save("bc_pretrained_checkpoint/")
    print(f"[BC] Saved pretrained checkpoint to {checkpoint_path}")

    return checkpoint_path