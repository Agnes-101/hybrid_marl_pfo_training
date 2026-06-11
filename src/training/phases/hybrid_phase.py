import sys
import os

from src.utils import kpi_logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
# print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

from logging import config
from typing import Dict
import numpy as np

from src.training.phases.marl_phase import execute_marl_phase
from src.training.phases.metaheuristic_phase import compare_algorithms, execute_metaheuristic_phase
from src.utils import kpi_logger
from src.training.phases.checkpoint import save_checkpoint


def get_initial_solution(config, env, current_epoch) -> Dict:
    """Return a solution dict, either from a single metaheuristic
    or by comparing all configured algorithms (if metaheuristic_comparison_mode)."""
    if config.get("metaheuristic_comparison_mode", False):
        algorithm_results = compare_algorithms(config, env, current_epoch)
        best_algorithm = max(
            algorithm_results,
            key=lambda x: algorithm_results[x]["metrics"]["fitness"]
        )
        print(f"\nBest algorithm selected: {best_algorithm.upper()}")
        print(f"Best algorithm metrics: {algorithm_results[best_algorithm]}")
        return algorithm_results[best_algorithm]
    else:
        return execute_metaheuristic_phase(env, current_epoch, algorithm=config["metaheuristic"])
    
def adaptive_retuning_required(config, kpi_logger) -> bool:
        """Check if metaheuristic retuning is needed"""
        metrics = kpi_logger.get_recent_metrics(
            window_size=config["adaptive_tuning"]["stagnation_window"]
        )
        return (np.mean(metrics["reward"]) < 
                config["adaptive_tuning"]["stagnation_threshold"])

def run_hybrid_loop(config, env, marl_algo, current_epoch, obs_space, act_space, kpi_logger) -> Dict:
    """Run a single epoch of the hybrid training loop, returning MARL results and metaheuristic metrics.
    This function encapsulates the core logic of one epoch, including:
    - Obtaining the initial solution (with optional comparison)
    - Executing the MARL phase with the obtained solution
    - Logging results to the KPI logger
    - Checking for adaptive retuning needs
    """

    initial_solution = get_initial_solution(config, env, current_epoch)
    print(f"Initial Solution is: {initial_solution}")

    for epoch in range(1, config["max_epochs"] + 1):
        current_epoch = epoch
        print(f"\n=== EPOCH {epoch} ===")
        # print(f'Marl_algo: {marl_algo}')
        # Execute MARL phase with direct algorithm control
        marl_results = execute_marl_phase( config, marl_algo, obs_space, act_space, kpi_logger, current_epoch=current_epoch )

        if kpi_logger:
            kpi_logger.log_epoch(
                epoch=epoch,
                marl_metrics={
                    "best_reward": marl_results["best_reward"],
                    "final_reward": marl_results["final_reward"],
                    "iterations": marl_results["iterations_completed"],
                },
                metaheuristic_metrics=initial_solution["metrics"],
            )

        if (config["adaptive_tuning"]["enabled"] and adaptive_retuning_required(config,kpi_logger)):
            print("Performance stagnation detected — retuning")
            initial_solution = get_initial_solution(config, env, current_epoch)  # respects comparison_mode

        if epoch % config["checkpoint_interval"] == 0:
            save_checkpoint(config, env, current_epoch)