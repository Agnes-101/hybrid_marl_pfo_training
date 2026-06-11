import sys
import os
from typing import Dict

from src.optimization.metaheuristic_opt import run_metaheuristic
from src.utils import kpi_logger
from src.utils.kpi_logger import KPITracker

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root)if project_root not in sys.path else None
# print(f"Verified Project Root: {project_root}")  # Should NOT be "/"

def execute_metaheuristic_phase(env,current_epoch,algorithm: str, kpi_logger=KPITracker(enabled=True)) -> Dict:
    """Run a single metaheuristic optimization"""
    print(f"\n Initializing {algorithm.upper()} optimization...")    
    
        # Pass the visualization hook to the metaheuristic
    solution = run_metaheuristic(
            env,
            algorithm,
            current_epoch,
            kpi_logger=kpi_logger, # Proper data flow
            visualize_callback= None # Proper data flow
        )
        
    print("Final KPI History after metaheuristic phase:")
        
        # Log and visualize results
    kpi_logger.log_algorithm_performance(algorithm=algorithm,metrics=solution["metrics"])
        # self.dashboard.update_algorithm_metrics(algorithm=algorithm,metrics=solution["metrics"] )        
    return solution

def compare_algorithms(config, env, current_epoch) -> Dict:
    """Run and compare multiple metaheuristics"""
    algorithm_results = {}      

    for algo in config["metaheuristic_algorithms"]:
        env.reset()
        algorithm_results[algo] = execute_metaheuristic_phase(env, current_epoch, algo) 
    print("\n=== Algorithm Comparison Results ===",)      
    return algorithm_results