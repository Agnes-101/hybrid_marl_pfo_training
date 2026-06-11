import sys
import os

from src.training.marl.ppo_builder import build_algorithm
from src.utils import kpi_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
# print(f"Verified Project Root: {project_root}")  # Should NOT be "/"


def execute_baseline_marl(config, kpi_logger=None,phase_name: str = "baseline_marl", marl_algo=None):
    """Execute baseline MARL training without metaheuristic initialization"""
    print("\n" + "="*50)
    print("BASELINE MARL TRAINING (No Hybrid)")
    print("="*50)
        
    # Use random or default initialization
    # baseline_results = execute_marl_phase(
    #         initial_policy=None,  # No metaheuristic initialization
    #         bc_checkpoint=None,    # no BC for baseline
    #         phase_name="baseline_marl"
    #     )
    baseline_results = execute_marl_phase(config,
            marl_algo=marl_algo,  # No pre-built algorithm for baseline
            phase_name=phase_name,
            kpi_logger=kpi_logger,
            current_epoch=0
        )
        
    return baseline_results
    
def execute_marl_phase(config, marl_algo, obs_space, act_space, kpi_logger, phase_name: str = "hybrid", current_epoch: int = 0):
    print(f"\nStarting {config.get('marl_algorithm','PPO').upper()} training ({phase_name})...")
        
        # Build/rebuild algorithm with new initial policy
        # self._build_algorithm(initial_policy)
        # We assume `algo` has already been built (and restored from BC if needed).
    algo = build_algorithm(config, obs_space, act_space, marl_algo)  # keep a reference so that stopping later is easy
        
        # Training metrics storage
    training_results = []
    best_reward = float('-inf')
        
        # Training loop using algo.train()
    for iteration in range(config["marl_steps_per_phase"]):
            # Single training step
        result = algo.train()
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

            
        #policy_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("policy_loss", 0)
        #vf_loss = result.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {}).get("vf_loss", 0)
            
        # Log detailed metrics to KPI Logger
        if kpi_logger:
            kpi_logger.log_iteration(
                    epoch=current_epoch,
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
        if iteration % 5 == 0 or iteration == config["marl_steps_per_phase"] - 1:
            print(f"  Iteration {iteration + 1}/{config['marl_steps_per_phase']}: "
                    f"Reward={episode_reward_mean:.2f}, "
                    f"Episode Length={episode_len_mean:.1f}, "
                    f"Policy Loss={policy_loss:.4f}")
            kpi_logger.save_to_csv()
            
            # Track best performance
        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
        # Early stopping based on performance
        if (config.get("early_stopping", {}).get("enabled", False) and
            iteration > config.get("early_stopping", {}).get("min_iterations", 20)):
                
            recent_rewards = [r.get("env_runners", {}).get("episode_reward_mean", 0) 
                                for r in training_results[-10:]]
            if len(recent_rewards) >= 10:
                improvement = max(recent_rewards) - min(recent_rewards)
                if improvement < config.get("early_stopping", {}).get("threshold", 0.01):
                    print(f"  Early stopping at iteration {iteration + 1} due to convergence")
                    break
        
        # Log phase summary to KPI Logger
    if kpi_logger:
        kpi_logger.log_phase_summary(
            epoch=current_epoch,
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
            "algorithm": algo  # Return algorithm instance for potential checkpoint saving
        }
        
    return final_stats