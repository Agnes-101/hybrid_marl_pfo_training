#hybrid_training.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"


# ENVIRONMENT REGISTRATION MUST be outside class definition
from ray.tune.registry import register_env
from src.envs.custom_channel_env import NetworkEnvironment, PolicyMappingManager

def env_creator(env_config):
    return NetworkEnvironment(env_config)

register_env("NetworkEnv", env_creator)


import ray
import time
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from ray import tune
from ray.rllib.models import ModelCatalog
# from ray.tune.trial import Trial
from typing import Dict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from src.analysis.comparison import MetricAnimator
from src.optimization.metaheuristic_opt import run_metaheuristic
from src.utils.kpi_logger import KPITracker
from src.utils.live_dashboard import LiveDashboard
# from ray.rllib.src.rl_module.rl_module import RLModule
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import gymnasium as gym
from collections import OrderedDict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import json

import numpy as np
import signal
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from collections import defaultdict

# 1) Seed all RNGs
np.random.seed(0)
torch.manual_seed(0)
   
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
    
# After MetaPolicy definition
ModelCatalog.register_custom_model("meta_policy", MetaPolicy)

import copy
import pickle

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


class VizCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Called at end of each training iteration."""
        it = result["training_iteration"]
        if it % 5 != 0:
            return

        env = algorithm.workers.local_worker().env
        policy = algorithm.get_policy()
        obs, _ = env.reset()
        associations = []

        for _ in range(env.episode_length):
            acts = {}
            for i in range(env.num_ue):
                o = obs[f"ue_{i}"]
                logits, _ = policy.model({"obs": o})
                acts[f"ue_{i}"] = int(logits.argmax(dim=-1).item())

            obs, _, done, _, _ = env.step(acts)
            associations.append([ue.associated_bs for ue in env.ues])
            if done["__all__"]:
                break

        # Dump JSON for Streamlit to pick up
        os.makedirs("/tmp/assoc", exist_ok=True)
        with open(f"/tmp/assoc/assoc_iter{it}.json", "w") as f:
            json.dump({"iteration": it, "associations": associations}, f)
        print(f"[VizCallback] wrote /tmp/assoc/assoc_iter{it}.json")
# RAY_DEDUP_LOGS=0
PYTHONWARNINGS="ignore::DeprecationWarning"
class HybridTraining:
    def __init__(self, config: Dict):
        # Initialize Ray AFTER path modification 
        # Initialize Ray with robust error handling
        try:
            if not ray.is_initialized():
                ray.init(
                    runtime_env={
                        "env_vars": {"PYTHONPATH": project_root},
                        "working_dir": project_root
                    },
                    logging_level=logging.INFO,
                    log_to_driver=True,
                    ignore_reinit_error=True,                    
                    **config.get("ray_resources", {})
                )
            
            # Try to verify packages are accessible but don't fail if they aren't
            try:
                @ray.remote
                def verify_package():
                    try:
                        # Print working directory and Python path for debugging
                        import os, sys
                        print(f"Current working directory: {os.getcwd()}")
                        print(f"Python path: {sys.path}")
                        
                        # Import required modules
                        from src.envs.custom_channel_env import NetworkEnvironment
                        print("✅ Successfully imported NetworkEnvironment")
                        return True
                    except ImportError as e:
                        print(f"❌ Import failed: {e}")
                        return False
                
                package_check = ray.get(verify_package.remote())
                if not package_check:
                    print("WARNING: Package verification failed, but continuing anyway...")
                
            except Exception as e:
                print(f"WARNING: Package verification error, but continuing: {e}")
        
        except Exception as e:
            print(f"Ray initialization error: {e}")
            # ray.init(
            #     ignore_reinit_error=True,
            #     num_cpus=2  # Minimal fallback configuration
            # )
        
        # Import NetworkEnvironment - handle both direct import and delayed import
        try:
            from src.envs.custom_channel_env import NetworkEnvironment
            self.env = NetworkEnvironment(config["env_config"])
        except ImportError:
            # Dynamic import as fallback
            import importlib
            try:
                module = importlib.import_module("src.envs.custom_channel_env")
                NetworkEnvironment = getattr(module, "NetworkEnvironment")
                self.env = NetworkEnvironment(config["env_config"])
            except Exception as e:
                raise ImportError(f"Failed to import NetworkEnvironment: {e}")   
        
        self.config = config
        self.env = NetworkEnvironment(config["env_config"])
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        self.kpi_logger = KPITracker(enabled=config["logging"]["enabled"])
        self.current_epoch = 0  # Track hybrid training epochs
        self.metaheuristic_runs = 0
        self.trainer = None
        self.max_metaheuristic_runs = 1          
        # Create log directory if needed
        if config["logging"]["enabled"]:
            os.makedirs(config["logging"]["log_dir"], exist_ok=True)       
        
            
            
        # Initialize algorithm instance as None
        self.algo = None
        signal.signal(signal.SIGINT, self.on_interrupt)

            
        print("Network topology and policy manager initialized")
        # print("Initial policy distribution:")
        self.env.log_policy_status()
        # 2) Grab the manager out of the env
        self.manager = self.env.policy_manager

        # 3) Define your mapping function *as a closure* over self.manager
        def policy_mapping_fn(agent_id, episode=None, **kwargs):
            bs_idx = self.manager.get_closest_bs(agent_id)
            return f"bs_{bs_idx}_policy"

        # 4) Store it as an attribute so you can pass it into PPOConfig later
        self.policy_mapping_fn = policy_mapping_fn
        
    def on_interrupt(self, sig, frame):
            print("Training interrupted (SIGINT). Saving current state…")
            self.kpi_logger.save_to_csv()
            sys.exit(0)  
            
    # def build_bc_dataset(self, num_samples: int):
    #     """
    #     Build a BC dataset of (obs_dict, action_dict) pairs by:
    #      1) bc_reset(seed) to get a fresh random state
    #       2) run the metaheuristic on that state
    #       3) capture obs_dict = env._get_obs()
    #       4) store (obs_dict, action_dict)
    #       5) repeat until num_samples is reached
    #     """
    #     dataset = []
    #     samples_collected = 0
    #     base_seed = 42

    #     while samples_collected < num_samples:
    #         # 1) Reset env to a new random state, using a fresh seed each iteration
    #         seed_for_this_episode = base_seed + samples_collected
    #         obs_dict, _ = self.env.bc_reset(seed=seed_for_this_episode)
    #         #    bc_reset(seed) must reseed RNG and randomize UE positions/demands,
    #         #    then return the new obs_dict via env._get_obs() internally.

    #         # 2) Run the metaheuristic to get an “initial_solution”
    #         #    length=num_ue list of BS indices in {0..num_bs-1}
    #         print(f"Samples Collected {samples_collected}...")
    #         meta_solution = self._execute_metaheuristic_phase(
    #             self.config["metaheuristic"]
    #         )
    #         initial_solution=meta_solution["solution"]
    #         # 3) After metaheuristic runs, grab observations again
    #         obs_dict = self.env._get_obs()
    #         #    This is: { "ue_0": obs_vec0, ..., "ue_{num_ue-1}": obs_vec_{num_ue-1} }

    #         # 4) Convert the metaheuristic’s output into action_dict
    #         action_dict = {}
    #         for ue_idx, bs_idx in enumerate(initial_solution):
    #             action_dict[f"ue_{ue_idx}"] = int(bs_idx)

    #         # 5) Store a deepcopy of (obs_dict, action_dict)
    #         dataset.append((copy.deepcopy(obs_dict), copy.deepcopy(action_dict)))
    #         samples_collected += 1

    #         # We do NOT call env.step() here—next loop iteration will call bc_reset() again

    #    return dataset
    def build_bc_dataset(self, num_samples: int, 
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
            obs_dict, _ = self.env.bc_reset(seed=seed_for_this_episode)

            # 2) Run your metaheuristic and get initial_solution
            meta_out = self._execute_metaheuristic_phase(self.config["metaheuristic"])
            initial_solution = meta_out["solution"]

            # 3) Grab new observations
            obs_dict = self.env._get_obs()

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
    
    def run_bc_pretraining(self,
                           trainer,
                           num_bc_samples=500,
                           bc_epochs=5,
                           batch_size=128):
        """
        1) Build a BC dataset by running the metaheuristic on random states.
        2) BC‐train the shared policy head inside the RLlib MetaPolicy.
        3) Save a checkpoint of the pretrained policy network, return its path.
        """
        num_ue = self.env.num_ue

        # Step A: build the raw (obs_dict, action_dict) dataset
        bc_list = self.build_bc_dataset(num_samples=num_bc_samples)
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
    
    def _execute_metaheuristic_phase(self, algorithm: str) -> Dict:
        """Run a single metaheuristic optimization"""
        print(f"\n Initializing {algorithm.upper()} optimization...")     
    
        # Pass the visualization hook to the metaheuristic
        solution = run_metaheuristic(
            self.env,
            algorithm,
            self.current_epoch,
            kpi_logger=self.kpi_logger,
            visualize_callback= None # Proper data flow
        )
        
        print("Final KPI History after metaheuristic phase:")
        
        # Log and visualize results
        self.kpi_logger.log_algorithm_performance(algorithm=algorithm,metrics=solution["metrics"])
        # self.dashboard.update_algorithm_metrics(algorithm=algorithm,metrics=solution["metrics"] )        
        return solution           
    

    def _adaptive_retuning_required(self) -> bool:
        """Check if metaheuristic retuning is needed"""
        metrics = self.kpi_logger.get_recent_metrics(
            window_size=self.config["adaptive_tuning"]["stagnation_window"]
        )
        return (np.mean(metrics["reward"]) < 
                self.config["adaptive_tuning"]["stagnation_threshold"])
    
    def _compare_algorithms(self) -> Dict:
        """Run and compare multiple metaheuristics"""
        algorithm_results = {}      

        for algo in self.config["metaheuristic_algorithms"]:
            self.env.reset()
            algorithm_results[algo] = self._execute_metaheuristic_phase(algo)       
        return algorithm_results
    
    def _build_algorithm(self, initial_policy: dict = None, bc_checkpoint: str = None):
        """Build or rebuild the PPO algorithm instance"""
        
        # Prepare initial weights
        initial_weights = []
        if initial_policy is not None:
            initial_weights = initial_policy.tolist()
            assert len(initial_weights) == self.config["env_config"]["num_ue"]
        
        env_config = {
            **self.config["env_config"],
            "initial_assoc": initial_policy
        }
        
        # Get a single UE's observation space as the template
        single_ue_obs_space = self.obs_space.spaces["ue_0"]  # This is gym.spaces.Box(shape=(obs_dim,))
        single_ue_act_space = self.act_space.spaces["ue_0"]  # Same for action space

        # policies = {
        #     f"ue_{i}_policy": (None, single_ue_obs_space, single_ue_act_space, {})
        #     for i in range(self.config["env_config"]["num_ue"])
        #     }
        policies = {
        "default_policy": (
            None,                 # No custom class name→ use default by name
            single_ue_obs_space,
            single_ue_act_space,
            {}
            )
            }
        
        # Build PPO config with policy sharing
        marl_config = (
            PPOConfig()
            .environment("NetworkEnv", env_config=env_config)
            # Total rollout size per iteration =rollout_fragment_length × num_env_runners             
            .env_runners(
                rollout_fragment_length=20,  # Increased from 10 for better experience collection              
                num_env_runners=0,  # Quad core machine, 1 or 2
                sample_timeout_s=600
                ) 
                    
            .training(
                model={
                    "custom_model": "meta_policy",
                    "custom_model_config": {
                        "initial_weights": initial_weights,
                        "num_bs": self.config["env_config"]["num_bs"],
                        "num_ue": self.config["env_config"]["num_ue"],
                    }
                },
                gamma=0.99,
                lr=1e-4, # 1e-4 for 60
                # lr_schedule=[(0, 1e-4), (50, 3e-4), (100, 1e-4)],
                lr_schedule=[(0, 1e-4), (50, 2e-4), (100, 1e-4)],  # Less aggressive peak 
                # lr_schedule=[(0, 5e-5), (1000, 1e-4), (10000, 5e-4)],
                entropy_coeff=0.02, #0.01,
                entropy_coeff_schedule=[
                    (0, 0.05),      # High exploration initially
                    (50, 0.02),   # Reduce as training progresses
                    (100, 0.01),  # Low exploration for fine-tuning
                    ],
                kl_coeff=0.2,
                train_batch_size_per_learner=900, # 1800, #  for 60
                sgd_minibatch_size=100, # 90, # 180, # for 60
                num_sgd_iter=6, # 10, # 8 for 60
                clip_param=0.15, # 0.15 for 60
                
            )
            .multi_agent(
                policies=policies, 
                # policy_mapping_fn=lambda agent_id, *_: "shared_policy"
                # policy_mapping_fn=self.policy_mapping_fn,
            )
        )
        
        # Clean up previous algorithm if exists
        if self.algo is not None:
            self.algo.stop()
            
        # Build new algorithm
        self.algo = marl_config.build()
        # 6) If a BC checkpoint is provided, restore it now
        # 7) Restore from BC checkpoint if provided
        if bc_checkpoint is not None:
            print(f"Restoring from BC checkpoint: {bc_checkpoint}")
            self.algo.restore(bc_checkpoint)
            print("Successfully restored BC pretrained weights")
            
        return self.algo
    # def _execute_marl_phase(self, initial_policy: dict = None, phase_name: str = "hybrid"):
    def _execute_marl_phase(self, *, algo, phase_name: str = "hybrid"):
        print(f"\nStarting {self.config.get('marl_algorithm','PPO').upper()} training ({phase_name})...")
        
        # Build/rebuild algorithm with new initial policy
        # self._build_algorithm(initial_policy)
        # We assume `algo` has already been built (and restored from BC if needed).
        self.algo = algo  # keep a reference so that stopping later is easy
        
        # Training metrics storage
        training_results = []
        best_reward = float('-inf')
        
        # Training loop using algo.train()
        for iteration in range(self.config["marl_steps_per_phase"]):
            # Single training step
            result = self.algo.train()
            training_results.append(result)
            
            # Extract key metrics
            episode_reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
            episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)
            learner = result.get("info", {}).get("learner", {})

            policy_losses = []
            vf_losses = []

            for policy_id, stats in learner.items():
                learner_stats = stats.get("learner_stats", {})
                policy_losses.append(learner_stats.get("policy_loss", 0.0))
                vf_losses.append(learner_stats.get("vf_loss", 0.0))

            policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
            vf_loss = sum(vf_losses) / len(vf_losses) if vf_losses else 0.0

            
            # policy_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("policy_loss", 0)
            # vf_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("vf_loss", 0)
            
            # Log detailed metrics to KPI Logger
            if hasattr(self, 'kpi_logger'):
                self.kpi_logger.log_iteration(
                    epoch=getattr(self, 'current_epoch', 0),
                    iteration=iteration,
                    phase=phase_name,
                    metrics={
                        "iteration": iteration,
                        "episode_reward_mean": episode_reward_mean,
                        "episode_len_mean": episode_len_mean,
                        "policy_loss": policy_loss,
                        "vf_loss": vf_loss,
                        "episodes_this_iter": result.get("env_runners", {}).get("episodes_this_iter", 0),
                        "timesteps_total": result.get("timesteps_total", 0)
                    }
                )
            
            # Log progress
            if iteration % 5 == 0 or iteration == self.config["marl_steps_per_phase"] - 1:
                print(f"  Iteration {iteration + 1}/{self.config['marl_steps_per_phase']}: "
                    f"Reward={episode_reward_mean:.2f}, "
                    f"Episode Length={episode_len_mean:.1f}, "
                    f"Policy Loss={policy_loss:.4f}")
                self.kpi_logger.save_to_csv()
            
            # Track best performance
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                
            # Optional: Early stopping based on performance
            if (self.config.get("early_stopping", {}).get("enabled", False) and
                iteration > self.config.get("early_stopping", {}).get("min_iterations", 20)):
                
                recent_rewards = [r.get("env_runners", {}).get("episode_reward_mean", 0) 
                                for r in training_results[-10:]]
                if len(recent_rewards) >= 10:
                    improvement = max(recent_rewards) - min(recent_rewards)
                    if improvement < self.config.get("early_stopping", {}).get("threshold", 0.01):
                        print(f"  Early stopping at iteration {iteration + 1} due to convergence")
                        break
        
        # Log phase summary to KPI Logger
        if hasattr(self, 'kpi_logger'):
            self.kpi_logger.log_phase_summary(
                epoch=getattr(self, 'current_epoch', 0),
                phase=phase_name,
                summary={
                    "iterations_completed": len(training_results),
                    "best_reward": best_reward,
                    "final_reward": training_results[-1].get("env_runners", {}).get("episode_reward_mean", 0),
                    "convergence_rate": (best_reward - training_results[0].get("env_runners", {}).get("episode_reward_mean", 0)) / len(training_results) if training_results else 0,
                    "total_timesteps": sum(r.get("timesteps_total", 0) for r in training_results)
                }
            )
        
        # Log final policy distribution
        # print("Final policy distribution after MARL phase:")
        # Note: This will show the temp_env distribution, but actual training envs
        # will have their own mobility patterns
        
        # Return summary of training
        final_stats = {
            "iterations_completed": len(training_results),
            "best_reward": best_reward,
            "final_reward": training_results[-1].get("env_runners", {}).get("episode_reward_mean", 0),
            "training_results": training_results,
            "algorithm": self.algo  # Return algorithm instance for potential checkpoint saving
        }
        
        return final_stats
    
    def _save_checkpoint(self, epoch: int):
        """Save algorithm checkpoint"""
        if self.algo is not None:
            checkpoint_path = f"{self.config['checkpointdir']}/epoch{epoch}_marl"
            self.algo.save_checkpoint(checkpoint_path)
            print(f"MARL checkpoint saved to {checkpoint_path}")
            return checkpoint_path
        return None
    
    def _restore_checkpoint(self, checkpoint_path: str):
        """Restore algorithm from checkpoint"""
        if self.algo is not None:
            self.algo.restore(checkpoint_path)
            print(f"MARL checkpoint restored from {checkpoint_path}")
    
    def _execute_baseline_marl(self):
        """Execute baseline MARL training without metaheuristic initialization"""
        print("\n" + "="*50)
        print("BASELINE MARL TRAINING (No Hybrid)")
        print("="*50)
        
        # Use random or default initialization
        baseline_results = self._execute_marl_phase(
            initial_policy=None,  # No metaheuristic initialization
            bc_checkpoint=None,    # no BC for baseline
            phase_name="baseline_marl"
        )
        
        return baseline_results
    
    def run_comparison_study(self):
        """Run both hybrid and baseline approaches for comparison"""
        comparison_results = {
            "hybrid": [],
            "baseline": []
        }
        
        num_runs = self.config.get("comparison_runs", 3)
        
        for run in range(num_runs):
            print(f"\n{'='*60}")
            print(f"COMPARISON RUN {run + 1}/{num_runs}")
            print(f"{'='*60}")
            
            # # Initialize fresh KPI logger for this run
            # if hasattr(self, 'kpi_logger'):
            #     self.kpi_logger.start_new_run(run, "comparison")
            
            # Run hybrid approach
            print(f"\n--- HYBRID APPROACH (Run {run + 1}) ---")
            # 2a) Build a “dummy” PPO Trainer so that we have a MetaPolicy instance
            #     with the correct architecture. We still pass no BC here—
            #     we only use this Trainer to do BC pretraining.
            dummy_algo = self._build_algorithm(
                initial_policy=None,
                bc_checkpoint=None
            )

            # 2b) Run BC pretraining on that dummy Algo. This returns a checkpoint path
            bc_checkpoint = self.run_bc_pretraining(
                # env=self.env,        # the actual NetworkEnvironment
                trainer=dummy_algo,  # the RLlib Trainer just created above
                num_bc_samples=50, # self.config["bc_samples"],
                bc_epochs=8,  #self.config["bc_epochs"],
                batch_size= 128, # self.config["bc_batch_size"]
            )

            # 2c) Tear down the dummy Trainer (we only needed it for BC)
            dummy_algo.stop()
            self.algo = None

            # 2d) Build a *new* PPO Trainer that restores from the BC checkpoint
            hybrid_algo = self._build_algorithm(
                initial_policy=None,
                bc_checkpoint=bc_checkpoint
            )

            # 2e) Run PPO training for the hybrid phase
            hybrid_result = self._execute_marl_phase(
                algo=hybrid_algo,
                phase_name="hybrid_marl"
            )
            comparison_results["hybrid"].append(hybrid_result)

            # 2f) Clean up the hybrid Trainer
            hybrid_algo.stop()
            self.algo = None
            
            
            # Run baseline MARL first
            print(f"\n--- BASELINE MARL (Run {run + 1}) ---")
            baseline_result = self._execute_baseline_marl()
            comparison_results["baseline"].append(baseline_result)
            
            # Reset algorithm and environment state
            if self.algo is not None:
                self.algo.stop()
                self.algo = None
            
            # # Run hybrid approach
            # print(f"\n--- HYBRID APPROACH (Run {run + 1}) ---")
            #  # 2a) Run BC pretraining (offline) to get a checkpoint
            # bc_checkpoint = run_bc_pretraining(
            #     env=self.env_instance,    # or however you access your NetworkEnv
            #     trainer=self,             # pass "self" if your trainer has been built once
            #     num_bc_samples=self.config["bc_samples"],
            #     bc_epochs=self.config["bc_epochs"],
            #     batch_size=self.config["bc_batch_size"]
            # )
            
            # # 2b) Build a new PPO trainer by restoring BC weights
            # hybrid_algo = self._build_algorithm(
            #     initial_policy=None,      # we don’t need meta bias once BC is done
            #     bc_checkpoint=bc_checkpoint
            # )

            # # 2c) Run PPO training for the hybrid phase
            # hybrid_result = self._execute_marl_phase(
            #     initial_policy=None,      # no additional bias
            #     bc_checkpoint=bc_checkpoint,
            #     phase_name="hybrid_marl"
            # )
            
            # comparison_results["hybrid"].append(hybrid_result)           
                    
        # Analyze and report comparison results
        self._analyze_comparison_results(comparison_results)
        
        return comparison_results
    
    def _analyze_comparison_results(self, results):
        """Analyze and report comparison between hybrid and baseline approaches"""
        import numpy as np
        
        baseline_rewards = [r["best_reward"] for r in results["baseline"]]
        hybrid_rewards = [r["best_reward"] for r in results["hybrid"]]
        
        
        baseline_convergence = [r["iterations_completed"] for r in results["baseline"]]
        hybrid_convergence = [r["iterations_completed"] for r in results["hybrid"]]
        
        print(f"\n{'='*60}")
        print("COMPARISON ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        baseline_mean=np.mean(baseline_rewards)
        baseline_std=np.std(baseline_rewards)
        hybrid_mean=np.mean(hybrid_rewards)
        hybrid_std=np.std(hybrid_rewards)
        
        baseline_iterations=np.mean(baseline_convergence)
        hybrid_iterations=np.mean(hybrid_convergence)
        
        print(f"\nPerformance Comparison ({len(baseline_rewards)} runs):")
        print(f"Baseline MARL:")
        print(f"  Mean Reward: {np.mean(baseline_rewards):.2f} ± {np.std(baseline_rewards):.2f}")
        print(f"  Best Reward: {np.max(baseline_rewards):.2f}")
        print(f"  Worst Reward: {np.min(baseline_rewards):.2f}")
        
        print(f"\nHybrid Approach:")
        print(f"  Mean Reward: {np.mean(hybrid_rewards):.2f} ± {np.std(hybrid_rewards):.2f}")
        print(f"  Best Reward: {np.max(hybrid_rewards):.2f}")
        print(f"  Worst Reward: {np.min(hybrid_rewards):.2f}")
        
        improvement = ((np.mean(hybrid_rewards) - np.mean(baseline_rewards)) / np.mean(baseline_rewards)) * 100
        print(f"\nHybrid Improvement: {improvement:.2f}%")
        
        print(f"\nConvergence Comparison:")
        print(f"Baseline Iterations: {np.mean(baseline_convergence):.1f} ± {np.std(baseline_convergence):.1f}")
        print(f"Hybrid Iterations: {np.mean(hybrid_convergence):.1f} ± {np.std(hybrid_convergence):.1f}")
        
        # Statistical significance test (simple t-test)
        from scipy import stats
        try:
            t_stat, p_value = stats.ttest_rel(hybrid_rewards, baseline_rewards)
            print(f"\nStatistical Significance:")
            print(f"T-statistic: {t_stat:.3f}")
            print(f"P-value: {p_value:.3f}")
            print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        except ImportError:
            print("\nInstall scipy for statistical significance testing")
        
        # Win rate
        wins = sum(1 for h, b in zip(hybrid_rewards, baseline_rewards) if h > b)
        win_rate = wins / len(hybrid_rewards) * 100
        print(f"\nHybrid Win Rate: {win_rate:.1f}% ({wins}/{len(hybrid_rewards)} runs)")
        
        # Training progress over iterations
        # 1. Learning Curves Comparison (Fixed)
        plt.subplot(2, 3, 1)
        x_baseline = range(len(baseline_rewards))
        x_hybrid = range(len(hybrid_rewards))
        plt.plot(x_baseline, baseline_rewards, 'o-', label='Baseline MARL', alpha=0.7)
        plt.plot(x_hybrid, hybrid_rewards, 's-', label='Hybrid Approach', alpha=0.7)
        # Add confidence intervals if we have multiple runs
        if len(baseline_rewards) > 1:
            plt.axhline(y=baseline_mean, color='blue', linestyle='--', alpha=0.5)
            plt.axhline(y=hybrid_mean, color='orange', linestyle='--', alpha=0.5)
        plt.xlabel('Run Number')
        plt.ylabel('Best Episode Reward')
        plt.title('Performance Across Runs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        #  Performance Distribution (Box Plots)        
        # 2. Performance Distribution (Box Plots)        
        plt.subplot(2, 3, 2)
        data = [baseline_rewards, hybrid_rewards]
        box_plot = plt.boxplot(data, labels=['Baseline\nMARL', 'Hybrid\nApproach'], patch_artist=True)
        # Color the boxes
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        plt.ylabel('Final Performance')
        plt.title('Performance Distribution')
        plt.grid(True, alpha=0.3)
        
        # 3. Convergence Speed Analysis
        plt.subplot(2, 3, 3)
        plt.scatter(baseline_convergence, baseline_rewards, 
                alpha=0.6, label='Baseline', s=60, c='blue')
        plt.scatter(hybrid_convergence, hybrid_rewards, 
                alpha=0.6, label='Hybrid', s=60, c='red')
        plt.xlabel('Iterations to Convergence')
        plt.ylabel('Final Reward')
        plt.title('Convergence Speed vs Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        #Statistical Significance Visualization
        plt.subplot(2, 2, 4)
        # Paired comparison plot
        for i in range(len(baseline_rewards)):
            plt.plot([1, 2], [baseline_rewards[i], hybrid_rewards[i]], 
                    'o-', alpha=0.5, color='gray')
        plt.plot([1, 2], [np.mean(baseline_rewards), np.mean(hybrid_rewards)], 
                'o-', linewidth=3, markersize=10, color='red')
        plt.xticks([1, 2], ['Baseline', 'Hybrid'])
        plt.ylabel('Performance')
        plt.title('Paired Comparison (Each Line = One Run)')
        
        # 5. Improvement Histogram
        plt.subplot(2, 3, 5)
        improvements = [(h-b)/b*100 for h, b in zip(hybrid_rewards, baseline_rewards)]
        plt.hist(improvements, bins=min(10, len(improvements)), alpha=0.7, 
                edgecolor='black', color='green')
        plt.xlabel('Improvement (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Improvements')
        plt.axvline(0, color='red', linestyle='--', label='No Improvement', linewidth=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        plt.subplot(2, 3, 6)
        categories = ['Mean\nReward', 'Max\nReward', 'Min\nReward', 'Std\nReward']
        baseline_stats = [baseline_mean, np.max(baseline_rewards), 
                        np.min(baseline_rewards), baseline_std]
        hybrid_stats = [hybrid_mean, np.max(hybrid_rewards), 
                    np.min(hybrid_rewards), hybrid_std]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x_pos - width/2, baseline_stats, width, label='Baseline', 
                color='lightblue', alpha=0.8)
        plt.bar(x_pos + width/2, hybrid_stats, width, label='Hybrid', 
                color='lightcoral', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Statistical Summary')
        plt.xticks(x_pos, categories)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        
        if not hasattr(self, 'kpi_logger') or not hasattr(self.kpi_logger, 'history'):
            print("No KPI logger history found.")
            return

        # Aggregate data by phase
        metrics_by_phase = defaultdict(lambda: {
            "reward": [],
            "policy_loss": [],
            "vf_loss": [],
            "timestamps": [],
            "iterations": []
        })

        for entry in self.kpi_logger.history:
            phase = entry.get("phase", "unknown")
            metrics = entry.get("metrics", {})
            metrics_by_phase[phase]["reward"].append(metrics.get("episode_reward_mean", 0))
            metrics_by_phase[phase]["policy_loss"].append(metrics.get("policy_loss", 0))
            metrics_by_phase[phase]["vf_loss"].append(metrics.get("vf_loss", 0))
            metrics_by_phase[phase]["iterations"].append(entry.get("iteration", 0))
            metrics_by_phase[phase]["timestamps"].append(entry.get("timestamp", 0))

        plt.figure(figsize=(12, 8))

        # 1. Training Efficiency Plot
        plt.subplot(2, 2, 1)
        for phase, data in metrics_by_phase.items():
            # Convert timestamps to relative minutes
            start_time = data["timestamps"][0]
            times_minutes = [(t - start_time).total_seconds() / 60.0 for t in data["timestamps"]]
            cumulative_rewards = np.cumsum(data["reward"])
            plt.plot(times_minutes, cumulative_rewards, label=f'{phase.capitalize()}', alpha=0.7)
        plt.xlabel('Training Time (minutes)')
        plt.ylabel('Cumulative Reward')
        plt.title('Training Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Policy Loss Convergence
        plt.subplot(2, 2, 2)
        for phase, data in metrics_by_phase.items():
            plt.plot(data["iterations"], data["policy_loss"], label=f'{phase.capitalize()} Policy Loss', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Policy Loss')
        plt.title('Policy Learning Stability')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Value Function Loss (optional extra)
        plt.subplot(2, 2, 3)
        for phase, data in metrics_by_phase.items():
            plt.plot(data["iterations"], data["vf_loss"], label=f'{phase.capitalize()} VF Loss', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Value Function Loss')
        plt.title('VF Loss Convergence')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


        
        return {
            "baseline_mean": np.mean(baseline_rewards),
            "hybrid_mean": np.mean(hybrid_rewards),
            "improvement_percent": improvement,
            "win_rate": win_rate,
            "statistical_significant": p_value < 0.05 if 'p_value' in locals() else None
        }
    
        
    def run(self, mode="hybrid"):
        """
        Main training loop with multiple modes
        
        Args:
            mode: "hybrid" for hybrid training, "baseline" for MARL only, 
                  "comparison" for running both approaches
        """
        if mode == "comparison":
            return self.run_comparison_study()
        elif mode == "baseline":
            return self._execute_baseline_marl()
        
        # Default hybrid mode
        try:
            # Initial metaheuristic phase (your existing code)
            if self.config["comparison_mode"]:
                algorithm_results = self._compare_algorithms()
                best_algorithm = max(
                    algorithm_results,
                    key=lambda x: algorithm_results[x]["metrics"]["fitness"]
                )
                print(f"\nBest algorithm selected: {best_algorithm.upper()}")
                initial_solution = algorithm_results[best_algorithm]
            else:
                if self.metaheuristic_runs < self.max_metaheuristic_runs:
                    initial_solution = self._execute_metaheuristic_phase(
                        self.config["metaheuristic"]
                    )
                    self.metaheuristic_runs += 1                
                print(f"Initial Solution is : {initial_solution}")
            
            # Hybrid training loop
            current_phase = "marl"
            print(f"\nCurrent phase: {current_phase}")
            
            for epoch in range(1, self.config["max_epochs"] + 1):
                self.current_epoch = epoch
                print(f"\n=== EPOCH {epoch} ===")
                
                if current_phase == "metaheuristic":
                    initial_solution = self._execute_metaheuristic_phase(
                        self.config["metaheuristic"]
                    )
                    print(f"Initial Solution is : {initial_solution}")
                    current_phase = "marl"
                
                # Execute MARL phase with direct algorithm control
                marl_results = self._execute_marl_phase(
                    initial_policy=initial_solution.get("solution"),
                    phase_name="hybrid_marl"
                )
                
                # Log hybrid performance (epoch-level summary)
                if hasattr(self, 'kpi_logger'):
                    self.kpi_logger.log_epoch(
                        epoch=epoch,
                        marl_metrics={
                            "best_reward": marl_results["best_reward"],
                            "final_reward": marl_results["final_reward"],
                            "iterations": marl_results["iterations_completed"]
                        },
                        metaheuristic_metrics=initial_solution["metrics"]
                    )
                
                # Adaptive phase switching
                if (self.config["adaptive_tuning"]["enabled"] and
                     self._adaptive_retuning_required()):
                    print("\nPerformance stagnation detected - triggering retuning")
                    current_phase = "metaheuristic"
                
                # Save system state and checkpoints
                if epoch % self.config["checkpoint_interval"] == 0:
                    # Save environment checkpoint
                    self.env.save_checkpoint(
                        f"{self.config['checkpointdir']}/epoch{epoch}.pkl"
                    )
                    # Save MARL algorithm checkpoint
                    self._save_checkpoint(epoch)
                    
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            # Clean up algorithm resources
            if self.algo is not None:
                self.algo.stop()
                print("Algorithm resources cleaned up")
    
    def evaluate_policy(self, num_episodes: int = 10):
        """Evaluate current policy performance"""
        if self.algo is None:
            print("No algorithm available for evaluation")
            return None
            
        print(f"Evaluating policy over {num_episodes} episodes...")
        
        # You can implement custom evaluation logic here
        # For now, just return recent training performance
        # In practice, you'd run the policy on test environments
        
        eval_results = {
            "episodes": num_episodes,
            "mean_reward": "TBD - implement evaluation logic",
            "std_reward": "TBD - implement evaluation logic"
        }
        
        return eval_results


if __name__ == "__main__":    
    config = {
        # Core configuration
        "metaheuristic": "pfo",
        "comparison_mode": False,
        "metaheuristic_algorithms": ["pfo", "co", "coa", "do", "fla", "gto", "hba", "hoa", "avoa","aqua", "poa", "rime", "roa", "rsa", "sto"], #
        "marl_algorithm": "PPO", 
        
        # Environment parameters
        "env_config": {
            "num_bs": 4,
            "num_ue": 30,
            "episode_length": 30,
            "log_kpis": True
        },
        
        "marl_steps_per_phase": 10,
        "max_epochs": 10,
        "checkpoint_interval": 5,
        "checkpointdir": "./checkpoints",
        "comparison_runs": 1,  # Number of runs for statistical comparison
        "early_stopping": {
            "enabled": True,
            "min_iterations": 30,
            "threshold": 0.01
        },
        
        # Resource management
        "ray_resources": {
            "num_cpus": 8,
            "num_gpus": 1,
            },
        "marl_training": {
        "num_gpus": 1  #  GPUs allocated to MARL
        },
        # Visualization parameters
        "visualization": {
            # "update_interval_ms": 500,
            "3d_resolution": 20,
            "heatmap_bins": 15
        },
        
        # Adaptive control
        "adaptive_tuning": {
            "enabled": False,
            "stagnation_threshold": -50,
            "stagnation_window": 10
        },
        
        # Logging
        "logging": {
            "enabled": True,
            "log_dir": "results/logs"
        }
    }

    trainer = HybridTraining(config)
    # trainer.run()
    comparison_results = trainer.run(mode="comparison")
