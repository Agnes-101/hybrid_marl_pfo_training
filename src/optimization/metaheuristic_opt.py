import sys
import os

# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))# ".."
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np
from typing import Dict, Any
from src.envs.custom_channel_env import NetworkEnvironment
from src.utils.kpi_logger import KPITracker
# from algorithms import aco, bat, cs, de, fa, ga, gwo, hs, ica, pfo, pso, sa, tabu, woa

# Core Metaheuristic Algorithms

from src.algorithms.coa import COAOptimization
from src.algorithms.co import CheetahOptimization
from src.algorithms.do import DandelionOptimization
from src.algorithms.gto import GTOOptimization
from src.algorithms.hba import HBAOptimization
from src.algorithms.rsa import RSAOptimization
from src.algorithms.sto import STOOptimization
from src.algorithms.poa import PelicanOptimization
from src.algorithms.hoa import HippoOptimization
from src.algorithms.fla import FLAOptimization
from src.algorithms.rime import RIMEOptimization
from src.algorithms.avoa import AVOAOptimization
from src.algorithms.aqua import AquilaOptimization
from src.algorithms.pfo import PolarFoxOptimization
from src.algorithms.roa import RainbowOptimization

# from src.algorithms.old.aco import ACOOptimization
# from src.algorithms.old.bat import BatOptimization
# from src.algorithms.old.cs import CSOptimization
# from src.algorithms.old.de import DEOptimization
# from src.algorithms.old.fa import FireflyOptimization
# from src.algorithms.old.ga import GAOptimization
# from src.algorithms.old.gwo import GWOOptimization
# from src.algorithms.old.hs import HarmonySearchOptimization
# from src.algorithms.old.ica import ICAOptimization
# from src.algorithms.pfo import PolarFoxOptimization
# from src.algorithms.old.pso import PSOOptimization
# from src.algorithms.old.sa import SAOptimization
# from src.algorithms.old.tabu import TabuSearchOptimization
# from src.algorithms.old.woa import WOAOptimization

from functools import partial
import json  # For serialize_result (add this)
# In your run_metaheuristic_task function
# from webapp.backend.tasks import web_visualize



    
def serialize_result(result: dict) -> dict:
    """Recursively convert Tensors/NumPy to JSON-safe types"""
    def _convert(obj):
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, int, float)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert(v) for v in obj]
        elif callable(obj):
            return f"Method: {obj.__name__}"  # Convert methods to strings for debugging
        return obj
    return _convert(result)

def run_metaheuristic(env: NetworkEnvironment, algorithm: str, epoch: int, kpi_logger: KPITracker,visualize_callback=None, iterations=1) -> dict:
    """
    Runs the selected metaheuristic algorithm and returns the optimized solution along with KPIs.
    
    :param algorithm: (str) Name of the metaheuristic algorithm to run.
    :param num_bs: (int) Number of base stations.
    :param num_ue: (int) Number of users.
    :return: Dict containing 'solution', 'SINR', 'fairness', 'load_balance', and 'handover_rate'.
    """
    # algorithms = {
    #     "aco": ACOOptimization,
    #     "bat": BatOptimization,
    #     "cs": CSOptimization,
    #     "de": DEOptimization,
    #     "fa": FireflyOptimization,
    #     "ga": GAOptimization,
    #     "gwo": GWOOptimization,
    #     "hs": HarmonySearchOptimization,
    #     "ica": ICAOptimization,
    #     "pfo": PolarFoxOptimization,
    #     "pso": PSOOptimization,
    #     "sa": SAOptimization,
    #     "tabu": TabuSearchOptimization,
    #     "woa": WOAOptimization,
    # }
    algorithms = {
                
        # Bio-inspired Algorithms
        "coa": COAOptimization,       # Coati Optimization Algorithm
        "do": DandelionOptimization,         # Dandelion Optimizer
        "gto": GTOOptimization,       # Gorilla Troops Optimizer
        "hba": HBAOptimization,       # Honey Badger Algorithm
        "rsa": RSAOptimization,       # Reptile Search Algorithm
        "sto": STOOptimization,       # Siberian Tiger Optimization
        "poa": PelicanOptimization,       # Pelican Optimization Algorithm
        "pfo": PolarFoxOptimization,
        "hoa": HippoOptimization,       # Hippopotamus Optimization Algorithm
        
        # Physics/Chemistry-inspired
        "fla": FLAOptimization,       # Fick's Law Algorithm
        "rime": RIMEOptimization,     # RIME Algorithm
        
        # Specialized Metaheuristics
        "avoa": AVOAOptimization,     # African Vultures Optimization Algorithm
        "co": CheetahOptimization, # Cheetah Optimizer
        "roa": RainbowOptimization, # Rainbow Optimization Algorithm
        "aqua": AquilaOptimization,
        # Add other algorithms as needed...
    }
    if algorithm not in algorithms:
        raise ValueError(f"Invalid metaheuristic algorithm '{algorithm}'. Choose from: {list(algorithms.keys())}")

    print(f"Running {algorithm.upper()} for initial optimization...")
    # Modified line: pass the environment to the run method
    # Instantiate with required parameters
    algo_class = algorithms[algorithm]
    algo_instance = algo_class(env=env,iterations=iterations,kpi_logger=kpi_logger)  # ✅ Pass logger

    print(f"\n Algorithm Instance, {algo_instance}")
    # solution = algo_instance.run(env)
    
    
    solution_data = algo_instance.run(
        visualize_callback=visualize_callback,
        kpi_logger=kpi_logger
    )

    # solution_data = algo_instance.run(
    #     # env=env,
    #     visualize_callback=  visualize_callback, #  web_visualize, ## Critical
    #     kpi_logger=kpi_logger
    # )
    solution= solution_data.get('solution')
    print(f"\n Algorithm Solution, {solution}")
    
    # Extract solution from the result
    # solution = solution_result["solution"]
    # solution = algorithms[algorithm].run(num_bs=num_bs, num_ue=num_ue).run(env)
    
    
    # # Evaluate using environment's built-in method
    metrics = env.evaluate_detailed_solution(solution)
    # print(f"\n Metrics, {metrics}")
    
    
    # # Log metrics using KPI tracker
    # kpi_logger.log_kpis(
    #     episode=epoch,
    #     reward=metrics["fitness"],
    #     sinr=metrics["average_sinr"],
    #     fairness=metrics["fairness"],
    #     load_variance=metrics["load_variance"]
    # )
    
    return {
        'algorithm': algorithm,
        "solution": solution,
        "metrics": metrics
        
    
    }
    # In return statement:
    # return {
    #     "solution": solution.tolist() if isinstance(solution, np.ndarray) else solution,
    #     "metrics": {
    #         k: v.tolist() if isinstance(v, np.ndarray) else v 
    #         for k, v in metrics.items()
    #     },
    #     "agents": {
    #         "positions": algo_instance.positions.tolist(),  # Explicit conversion
    #         "fitness": algo_instance.best_fitness
    #     }
    # }


def get_agent_states(algo_instance: Any) -> Dict:
    """Universal agent state extractor for all algorithms"""
    # Try common attribute patterns
    positions = _get_possible_attribute(
        algo_instance, 
        ["positions", "population", "swarm", "spatial_positions"]
    )
    
    fitness = _get_possible_attribute(
        algo_instance,
        ["fitness", "fitness_history", "best_fitnesses"]
    )
    
    # Special cases (e.g., ACO pheromones)
    trails = getattr(algo_instance, "pheromone_trails", None)
    
    # return {
    #     "positions": _normalize_positions(positions),
    #     "fitness": _normalize_fitness(fitness),
    #     "trails": trails
    # }
    return {
        "positions": algo_instance.positions.tolist(),
        "fitness": float(algo_instance.best_fitness),  # Ensure scalar
        "iteration": int(algo_instance.curr_iteration)
    }

def _get_possible_attribute(obj, attr_names):
    """Safely get first matching attribute"""
    for name in attr_names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None  # Or default array

def _normalize_positions(positions):
    """Convert any position format to (x, y, fitness)"""
    if positions is None:
        return np.empty((0, 3))
    
    # Handle different formats
    if isinstance(positions, list):  # Common in population-based algorithms
        return np.array([[p[0], p[1], f] for p, f in zip(positions, fitness)])
    
    return positions  # Assume already formatted

def _normalize_fitness(fitness):
    """Ensure fitness is a numpy array"""
    return np.array(fitness) if fitness is not None else np.array([])

# def get_agent_states(algorithm_instance: Any) -> Dict:
#     """Universal agent state extractor based on algorithm properties"""
#     return {
#         "positions": getattr(algorithm_instance, 'positions', np.empty((0, 2))),
#         "fitness": getattr(algorithm_instance, 'fitness', np.array([])),
#         "velocity": getattr(algorithm_instance, 'velocity', None),  # Optional
#         "trails": getattr(algorithm_instance, 'pheromone_trails', None)  # For ACO
#     }  

    
