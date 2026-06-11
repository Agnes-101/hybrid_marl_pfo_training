import sys
import os

from src.utils import kpi_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.utils import kpi_logger
from .comp_plots import *

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
                    
        # Analyze and report comparison results
    self._analyze_comparison_results(comparison_results)
    
    return comparison_results
    
def analyze_comparison_results(self, results):
    """Analyze and report comparison between hybrid and baseline approaches"""
    
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
    plot_learning_curves_comparison(baseline_rewards, hybrid_rewards, baseline_mean, hybrid_mean)    
        
    #  Performance Distribution (Box Plots) 
    plot_performance_distribution_comparison(baseline_rewards, hybrid_rewards)

    plot_convergence_speed_comparison(baseline_convergence, hybrid_convergence, baseline_rewards, hybrid_rewards)
    plot_paired_comparison(baseline_rewards, hybrid_rewards)
    plot_improvement_histogram(baseline_rewards, hybrid_rewards)
    generate_comparison_plots(baseline_rewards, hybrid_rewards, baseline_mean, hybrid_mean, baseline_std, hybrid_std)
        
    if not hasattr('kpi_logger') or not hasattr(kpi_logger, 'history'):
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
    plot_training_efficiency(metrics_by_phase)
    plot_policy_loss_convergence(metrics_by_phase)
    plot_value_function_loss_convergence(metrics_by_phase)
    plt.tight_layout()
    plt.show()


        
    return {
            "baseline_mean": np.mean(baseline_rewards),
            "hybrid_mean": np.mean(hybrid_rewards),
            "improvement_percent": improvement,
            "win_rate": win_rate,
            "statistical_significant": p_value < 0.05 if 'p_value' in locals() else None
        }




