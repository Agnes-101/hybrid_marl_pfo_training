import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

from ray.rllib.algorithms.ppo import PPOConfig


def build_algorithm( config, obs_space, act_space, marl_algo, initial_policy: dict = None, bc_checkpoint: str = None):
    """Build or rebuild the PPO algorithm instance"""
        
    # Prepare initial weights
    initial_weights = []
    if initial_policy is not None:
        initial_weights = initial_policy.tolist()
        assert len(initial_weights) == config["env_config"]["num_ue"]
        
    env_config = {
            **config["env_config"],
            "initial_assoc": initial_policy
        }
        
        # Get a single UE's observation space as the template
    single_ue_obs_space = obs_space.spaces["ue_0"]  # This is gym.spaces.Box(shape=(obs_dim,))
    single_ue_act_space = act_space.spaces["ue_0"]  # Same for action space

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
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
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
                        "num_bs": config["env_config"]["num_bs"],
                        "num_ue": config["env_config"]["num_ue"],
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
                # sgd_minibatch_size=100, # 90, # 180, # for 60
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
    if marl_algo is not None:
        marl_algo.stop()
            
    # Build new algorithm
    marl_algo = marl_config.build()
        # 6) If a BC checkpoint is provided, restore it now
        # 7) Restore from BC checkpoint if provided
    if bc_checkpoint is not None:
        print(f"Restoring from BC checkpoint: {bc_checkpoint}")
        marl_algo.restore(bc_checkpoint)
        print("Successfully restored BC pretrained weights")
            
    return marl_algo