#hybrid_training.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

from src.training.phases.hybrid_phase import run_hybrid_loop
from configs.configs_loader import load_config

config = load_config("default_configs")

# ENVIRONMENT REGISTRATION MUST be outside class definition
from ray.tune.registry import register_env
from src.envs.custom_channel_env import NetworkEnvironment
# from src.analysis.comparison import MetricAnimator
from src.utils.kpi_logger import KPITracker
from src.utils.live_dashboard import LiveDashboard
from src.training.marl.meta_policy import MetaPolicy
from src.training.bc.bc_pretraining import run_bc_pretraining
from src.training.callbacks.viz_callbacks import VizCallback
from src.training.phases.metaheuristic_phase import execute_metaheuristic_phase, compare_algorithms
from src.training.phases.marl_phase import execute_baseline_marl, execute_marl_phase
from src.training.phases.checkpoint import save_checkpoint, restore_checkpoint
from src.training.analysis.comparison_report import run_comparison_study


def env_creator(env_config):
    return NetworkEnvironment(env_config)

register_env("NetworkEnv", env_creator)


import ray
import logging
import torch
import numpy as np
import threading

from ray.rllib.models import ModelCatalog
# from ray.tune.trial import Trial
from typing import Dict
import numpy as np
import signal
import pandas as pd


# 1) Seed all RNGs
np.random.seed(0)
torch.manual_seed(0)
   
    
# After MetaPolicy definition
ModelCatalog.register_custom_model("meta_policy", MetaPolicy)


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
                    # logging_level=logging.INFO,
                    # log_to_driver=True,
                    ignore_reinit_error=True,                    
                    **config.get("ray_resources", {})
                )
            
            # Try to verify packages are accessible but don't fail if they aren't
            try:
                @ray.remote
                def verify_package():
                    try:
                                 
                        # Import required modules
                        from src.envs.custom_channel_env import NetworkEnvironment
                        print(" Successfully imported NetworkEnvironment")
                        return True
                    except ImportError as e:
                        print(f"Import failed: {e}")
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
        self.marl_algo = None
        

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.on_interrupt)

            
        print("Network topology and policy manager initialized")
        # print("Initial policy distribution:")
        # self.env.log_policy_status()
        # 2) Grab the manager out of the env
        self.manager = self.env.policy_manager

        # 3) Define your mapping function *as a closure* over self.manager
        def policy_mapping_fn(agent_id, episode=None, **kwargs):
            bs_idx = self.manager.get_closest_bs(agent_id)
            return f"bs_{bs_idx}_policy"

        # 4) Store it as an attribute so you can pass it into PPOConfig later
        self.policy_mapping_fn = policy_mapping_fn
        
    def on_interrupt(self):
            print("Training interrupted (SIGINT). Saving current state…")
            self.kpi_logger.save_to_csv()
            sys.exit(0) 

    def run(self, training_mode="hybrid"):
        """
        Main training loop with multiple modes
        
        Args:
            mode: "hybrid" for hybrid training, 
                  "baseline" for MARL only, 
                  "comparison" for running both approaches
        """
        try:
            if training_mode == "hybrid_vs_baseline_marl":
                return run_comparison_study()
            elif training_mode == "baseline_marl":
                return execute_baseline_marl(self.config, self.kpi_logger, self.current_epoch, self.marl_algo, self.obs_space, self.act_space)
            elif training_mode == "metaheuristic_comparison":
                return compare_algorithms(self.config, self.env, self.current_epoch)
            else :
                return run_hybrid_loop(self.config, self.env, self.marl_algo, self.current_epoch, self.obs_space, self.act_space, self.kpi_logger)       
                                    
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            # Clean up algorithm resources
            if self.marl_algo is not None:
                self.marl_algo.stop()
                print("Algorithm resources cleaned up")


# if __name__ == "__main__":  
    
#     trainer = HybridTraining(config)
#     trainer.run(training_mode="baseline_marl")
