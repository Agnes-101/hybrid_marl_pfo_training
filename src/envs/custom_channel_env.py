        
import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import torch
import numpy as np
import gymnasium as gym
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, List
from utils.kpi_logger import KPITracker  # Import the KPI logger
import time
import math
import numpy as np

from envs.entities.ue import UE
from envs.entities.base_station import BaseStation
from src.envs.association_manager import AssociationMappingManager
from envs.geometry.topology import generate_hex_positions
from envs.env_state.snapshot import get_state_snapshot, set_state_snapshot
from envs.kpi_metrics import *
    
class NetworkEnvironment(MultiAgentEnv):       
    
    def __init__(self, config:EnvContext, log_kpis=True,seed=None):        
        super().__init__()  # Initialize gym.Env
        self.config = config
        # Define observation space first
        # Access passed environment instance
        self.num_bs = config.get("num_bs", 20)
        self.num_ue = config.get("num_ue", 200)
        self.episode_length = config.get("episode_length", 100)
        self.env_instance = config.get("environment_instance")
        
        self.version = 0  # Internal state version
        self.current_step = 0
        self.log_kpis = log_kpis        
        # self.metaheuristic_agents = []  # Initialize empty list 
        self.seed = seed if seed is not None else 42
        if self.seed is not None:
            np.random.seed(self.seed)
            
        self.step_count = 0 
        self.episode_counter = 0       
        self.base_stations = []
        # Calculating hexagonal positions for BS             
        
        # In NetworkEnvironment.__init__:
        grid = generate_hex_positions(
            num_bs=self.num_bs,
            min_distance_from_center=30.0,
            enforce_center=True
        )
        
        colors = ["A", "B", "C"]  # Simple 3-color reuse
        # Define bandwidths (aligned with 3GPP recommendations)
        MACRO_CONFIG = {
            "frequency": 2.1e9,    # Sub-6 GHz
            "bandwidth": 20e6,     # 20 MHz (typical macro cell)
            "tx_power_dbm": 46.0,
            "path_loss_n": 3.5
        }

        SMALL_CELL_CONFIG = {
            "A": {
                "frequency": 26.75e9,  # # full n258 BW 28 GHz
                "bandwidth": 3.25e9    # 2 GHz (mmWave typical)
            },
            "B": {
                "frequency": 28.00e9,  # # full n257 BW 39 GHz
                "bandwidth": 3.00e9  # 4 GHz 
            },
            "C": {
                "frequency": 39.00e9,  #  # full n260 BW 60 GHz
                "bandwidth": 3.00e9   # 8 GHz
            }
        }

        # Macro BS initialization
        macro_bs = BaseStation(
            id=0,
            position=grid[0],
            frequency=MACRO_CONFIG["frequency"],
            bandwidth=MACRO_CONFIG["bandwidth"],
            subcarrier_spacing=15e3,
            reuse_color="Macro",
            tx_power_dbm=MACRO_CONFIG["tx_power_dbm"],
            path_loss_n=MACRO_CONFIG["path_loss_n"],
            tx_gain_dbi=8.0,    # Standard cellular antenna
            bf_gain_dbi=10.0,    # Limited beamforming
            path_loss_sigma=8.0,
            cre_bias=0.0            
        )
        self.base_stations.append(macro_bs)
        # Small cells initialization
        small_cells = []
        for i, pos in enumerate(grid[1:]):  # Skip macro
            color = colors[i % len(colors)]
            config = SMALL_CELL_CONFIG[color]
            
            small_cells.append(BaseStation(
                id=i+1,
                position=pos,
                frequency=config["frequency"],
                bandwidth=config["bandwidth"],
                subcarrier_spacing=60e3,
                reuse_color=color,
                tx_power_dbm=30.0,
                path_loss_n=2.1,
                path_loss_sigma=4.0,
                cre_bias=6.0,
                tx_gain_dbi=12.0,  # High-gain mmWave antenna
                bf_gain_dbi=25.0,   # Advanced beamforming
                height=10.0         # Small cell height
            ))
        self.base_stations.extend(small_cells)
        all_positions = np.random.uniform(0, 100, size=(self.num_ue, 2)).astype(np.float32)
        self.ues = [
            UE(
                id=i,
                position=all_positions[i],
                demand=np.random.randint(5, 20),
                v_min=0.5,
                v_max=1.5,
                pause_min=1.0,
                pause_max=5.0,
                ewma_alpha=0.9
            )
            for i in range(self.num_ue)
        ]
        print(f"Created {len(self.ues)} UEs")
        # self.ue_positions = {}
        # for ue in self.ues:
        #     agent_id = f"ue_{ue.id}"  # Convert int ID to string format
        #     self.ue_positions[agent_id] = ue.position
        self.prev_associations = {ue.id: None for ue in self.ues}
        self.handover_counts  = {ue.id: 0    for ue in self.ues}
        self.load_history = {bs.id: [] for bs in self.base_stations}
        self.initial_assoc   = config.get("initial_assoc", None)
        self._has_warm_start = False
        for bs in self.base_stations:
            bs.base_stations = self.base_stations  # for interference loops
            bs.ues           = self.ues            # so data_rate_shared can see all UEs
        # Initialize KPI logger if logging is enabled
        self.kpi_logger = KPITracker() if log_kpis else None        
        # obs_dim = 3*self.num_bs + 1 + (self.num_bs + 1) # SINRs + BS loads + BS Utilizations + own demand + Connected
        # obs_dim = 3*self.num_bs + 3 + (self.num_bs + 1)
        obs_dim = 2*self.num_bs + 2
        self.observation_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,), dtype=np.float32
            )
            for i in range(self.num_ue)
        })

        self.action_space = gym.spaces.Dict({
            f"ue_{i}": gym.spaces.Discrete(self.num_bs)
            for i in range(self.num_ue)
        })

        self._initialize_policy_manager()
        
    def _initialize_policy_manager(self):
        """Initialize the policy mapping manager with current BS and UE positions"""
        # Extract BS positions
        bs_positions = np.array([bs.position for bs in self.base_stations])
        
        # Extract initial UE positions
        initial_ue_positions = {}
        for ue in self.ues:
            agent_id = f"ue_{ue.id}"
            initial_ue_positions[agent_id] = np.array(ue.position)
        
        # Create policy manager
        self.policy_manager = AssociationMappingManager(bs_positions, initial_ue_positions)
        
        # print(f"Policy manager initialized with {len(bs_positions)} BSs and {len(initial_ue_positions)} UEs")
        # print("Initial policy distribution:")
        self.policy_manager.log_association_assignments()
        
    def get_policy_for_agent(self, agent_id: str) -> str:
        """Get the policy name for a given agent based on current position"""
        closest_bs = self.policy_manager.get_closest_bs(agent_id)
        return f"bs_{closest_bs}_policy"
    
    def get_association_distribution(self) -> Dict[str, int]:
        """Get current policy distribution across all UEs"""
        return self.policy_manager.get_association_distribution()
    
        
    
    
    def reset(self, *, seed=None, options=None):
        # 1) Reset step counter and clear all BS loads & UE state
        self.current_step = 0
        for bs in self.base_stations:
            bs.allocated_resources.clear()
            bs.calculate_load()
        for ue in self.ues:
            ue.associated_bs = None
            ue.sinr          = -np.inf
            ue.ewma_dr       = 0.0

        # 2) ONE-TIME metaheuristic warm start
        if not self._has_warm_start and self.initial_assoc is not None:
            for ue in self.ues:
                bs_idx = self.initial_assoc[ue.id]
                ue.associated_bs = bs_idx
                dr = self.base_stations[bs_idx].data_rate_shared(ue)
                self.base_stations[bs_idx].allocated_resources[ue.id] = dr
            # Recompute each BS’s load once after all assignments
            for bs in self.base_stations:
                bs.calculate_load()

            # Mark that warm start has been applied
            self._has_warm_start = True
            # Optionally clear to free memory
            self.initial_assoc = None
            # Reinitialize policy manager with reset positions
        self._initialize_policy_manager()
        
        # 3) Build and return the obs + infos dicts
        obs   = self._get_obs()  
        infos = {f"ue_{i}": {} for i in range(self.num_ue)}
        return obs, infos
        # return self._get_obs()# , {}
          
    def calculate_individual_reward(self, agent_id=None):
        if agent_id is None:
            return 0.0

        # Parse UE index
        if isinstance(agent_id, str) and agent_id.startswith("ue_"):
            ue_id = int(agent_id.split("_")[1])
            ue = self.ues[ue_id]

            # 1) Unconnected penalty
            if ue.associated_bs is None:
                return -1.0

            # 2) SINR factor (linear → clipped & normalized)
            lin_snr = ue.sinr  # already linear
            # Clip to [0, SNR_max] and normalize
            SNR_MAX = 100.0
            snr_clipped = max(0.0, min(lin_snr, SNR_MAX))
            sinr_factor = snr_clipped / SNR_MAX  # ∈ [0,1]

            # 3) Load factor (PRB utilization)
            bs = self.base_stations[ue.associated_bs]
            prb_util = bs.load / (bs.num_rbs + 1e-9)  # ∈ [0,1]
            
            # load_factor = 1.0 - prb_util           # ∈ [0,1]
            
            optimal_util = 0.8  # Target 80% utilization
            if prb_util <= optimal_util:
                load_factor = prb_util / optimal_util  # Reward up to optimal
            else:
                # Penalize overload progressively
                overload_ratio = prb_util / optimal_util
                load_factor = max(0.0, 2.0 - overload_ratio)
            
            # 4) Handover penalty
            handover_penalty = 0.0
            if hasattr(ue, 'prev_associated_bs'):
                if ue.prev_associated_bs != ue.associated_bs:
                    handover_penalty = 0.1

            # 4) Composite reward
            #    weight SINR high if you care throughput, weight load high if you care fairness
            w_snr, w_load = 0.7, 0.3
            base_reward = w_snr * sinr_factor + w_load * load_factor

            # 5) Scale into a convenient range (e.g. [–1, +1])
            #    Here we map base_reward ∈ [0,1] → [0,+1], then shift down for unconnected
            return float(base_reward)

        return 0.0
        
    def calculate_global_reward(self):
        """
        Normalized reward: Gbps throughput + fairness - overload_penalty.
        
        Assumptions:
          - Each BS stores per-UE throughput in bits/s in bs.allocated_resources (dict ue_id->bits/s).
          - bs.capacity is in Mbps (set by calculate_capacity_rb_based()).
        """
        # 1) Total system throughput (bits/s) → Gbps
        total_bps = sum(
            dr
            for bs in self.base_stations
            for dr in bs.allocated_resources.values()
        )
        throughput_gbps = total_bps / 1e9

        # 2) Refresh each BS capacity (Mbps)
        for bs in self.base_stations:
            bs.capacity = bs.calculate_capacity_rb_based()
            # print(f"BS {bs.id}, Capacity: {bs.capacity}")
        # 3) Build load and capacity tensors (both in bps)
        loads_bps = torch.tensor(
            [sum(bs.allocated_resources.values()) for bs in self.base_stations],
            dtype=torch.float32
        )
        capacities_bps = torch.tensor(
            [bs.capacity * 1e6 for bs in self.base_stations],
            dtype=torch.float32
        )

        # 4) Normalized load per BS (unitless)
        loads_norm = loads_bps / (capacities_bps + 1e-9)

        # 5) Jain’s fairness index on normalized loads
        # #    J = (sum x_i)^2 / (N * sum x_i^2)
        # N = len(self.base_stations)
        # sum_loads = loads_norm.sum()
        # fairness = (sum_loads * sum_loads) / (N * (loads_norm * loads_norm).sum() + 1e-6)
        
        # Option 3: Load Balance Variance (simpler alternative)
        # Penalize high variance in utilization rates
        load_variance = torch.var(loads_norm)
        balance_score = 1.0 / (1.0 + load_variance)  # Higher is better        
        # 4) Overload penalty
        overload = torch.relu(loads_norm - 1.0).sum()
        
        # 5) Connection ratio bonus
        connected_ues = sum(1 for ue in self.ues if ue.associated_bs is not None)
        connection_ratio = connected_ues / len(self.ues)
        
        # 6) Normalize throughput
        max_expected_throughput = 10.0
        throughput_normalized = min(throughput_gbps / max_expected_throughput, 1.0)
        
        # reward = (
        #     1.0 * throughput_gbps   # reward raw capacity in Gbps
        #     + 2.0 * fairness        # weight fairness
        #     - 1.0 * overload        # penalize overloaded cells
        # )
        
        reward = (
            2.0 * throughput_normalized +
            2.0 * balance_score +
            1.0 * connection_ratio -
            3.0 * overload
        )
        return reward
    
    def step(self, actions):
        try:
            # print(f"Step called with {len(actions)} actions")
            start_time = time.time()            
            connected_count = 0
            # 1) Decode and apply associations
            connected_count = self._step_apply_actions(actions, connected_count)

            # 2) Run PF scheduler on each BS
            self._step_run_scheduler()

            # 3) Update SINR & EWMA
            self._update_system_metrics()
            # 4) Compute per-agent rewards
            rewards = self._step_compute_rewards()
            # print(f"Rewards type immediately: {type(rewards)}")

            step_time = time.time() - start_time
            
             # 5) Decide termination/truncation
            terminated , truncated = self._step_check_done()
            
            # print(f"Connected Users : {connected_count} Users")
            if truncated["__all__"] or terminated["__all__"]:               
                self._step_log_kpis(rewards, step_time)             
                self.episode_counter += 1
                # print(f"Reward type at episode end: {type(rewards)}")

            # 5) Build infos, check termination
            obs = self._get_obs()             
            per_agent_info = self._step_build_per_agent_info()
            
            # 3) Assemble common (__all__) info
            # print(f"Rewards type before common info: {type(rewards)}")
            common_info = self._step_compute_common_metrics(rewards, step_time)
            diagnostics = self._step_build_overall_diagnostics()
            common_info["diagnostics"] = diagnostics
            # # Create info dict with one entry per agent, plus global info
            
            info = per_agent_info
            info["__common__"] = common_info
            self.current_step += 1
            
            self._step_update_mobility()      
            
            return obs, rewards, terminated, truncated, info
        except Exception as e:        
            print(f"ERROR in step: {e}")
            import traceback
            print(traceback.format_exc())
            # Return a safe default response
            return self._get_obs(), {f"ue_{ue.id}": 0.0 for ue in self.ues}, {"__all__": False}, {"__all__": True}, {"__common__": {"error": str(e)}} 
        
    def _step_apply_actions(self, actions, connected_count):
        for agent_id, a in actions.items():
            idx = int(agent_id.split("_")[1])
            self.ues[idx].associated_bs = None if a == 0 else (a - 1)
            if a != 0:
                connected_count += 1
        return connected_count

    def _step_run_scheduler(self):
        for bs in self.base_stations:
            bs.ues = {ue.id: ue for ue in self.ues if ue.associated_bs == bs.id}
            bs.allocate_prbs() 
    def _step_compute_rewards(self):
        return {
            f"ue_{ue.id}": self.calculate_individual_reward(f"ue_{ue.id}")
            for ue in self.ues
        }
    def _step_check_done(self):
        terminated = {"__all__": False}
        truncated = {"__all__": self.current_step >= self.episode_length}
        return terminated, truncated
    def _step_compute_common_metrics(self, rewards, step_time):
        # 4b) Compute aggregate metrics for logging
        total_reward = sum(rewards.values())
                # Connected ratio
        connected_ratio = sum(1 for ue in self.ues if ue.associated_bs is not None) / self.num_ue

        # Load‐balancing fairness (Jain) on normalized PRB loads
        jains = compute_jains_fairness(self.base_stations)

        # Safe throughput sum (Gbps)
        total_throughput = 0.0
        for ue in self.ues:
            if ue.associated_bs is not None:
                lin = min(ue.sinr, 100.0)
                total_throughput += np.log2(1 + 10**(lin/10))
                        
        # now total_throughput is in bits/s per Hz—if you want Gbps, multiply by your PRB_bw and num_prbs:self.rb_bandwidth 
        total_throughput_gbps = total_throughput * 180e3 * len(self.base_stations[0].rb_allocation) / 1e9

        return {
            "connected_ratio": connected_ratio,
            "step_time_s": step_time,
            "avg_reward": (
                total_reward / self.num_ue
                if self.num_ue > 0
                else 0.0
            ),
            "total_throughput_Gbps": total_throughput_gbps,
            "fairness_index": float(jains)
        }
    def _step_build_overall_diagnostics(self):
        return {
            "current_solution": [
                ue.associated_bs
                if ue.associated_bs is not None
                else -1
                for ue in self.ues
            ],

            "sinr_list": [
                float(min(ue.sinr, 100.0))
                if ue.associated_bs is not None
                else -np.inf
                for ue in self.ues
            ]
        }
    def _step_build_per_agent_info(self):
        info = {}

        for ue in self.ues:
            connected = ue.associated_bs is not None

            sinr_db = (
                min(ue.sinr, 100.0)
                if connected
                else -np.inf
            )
        
            throughput = (
                np.log2(1 + 10 ** (sinr_db / 10))
                if connected
                else 0.0
            )

            info[f"ue_{ue.id}"] = {
                "connected": connected,
                "sinr_dB": float(sinr_db),
                "throughput_bps_per_hz": float(throughput),
            }
        return info
    def _step_update_mobility(self):
        # Update UE positions using RWP mobility model
        for ue in self.ues:
            ue.update_position()
        # Update policy manager with new positions
        new_positions = {}
        for ue in self.ues:
            agent_id = f"ue{ue.id}"
            new_positions[agent_id] = np.array(ue.position)
        self.policy_manager.update_ue_positions(new_positions)
    
    def _step_log_kpis(self, rewards, step_time):
        # print(type(rewards))
        # print(rewards)
        compute_metrics = self._step_compute_common_metrics(rewards, step_time)
        compute_diagnostics = self._step_build_overall_diagnostics()
        if not (self.log_kpis and self.kpi_logger):
            return

        # 4c) Log to KPI
        if self.log_kpis and self.kpi_logger:
            # print("Updating Metrics per Episode in Step....")
            metrics = {
                    "connected_ratio": compute_metrics["connected_ratio"],
                    "step_time": step_time,
                    "episode_reward_mean": compute_metrics["avg_reward"],
                    "fairness_index": compute_metrics["fairness_index"],
                    "total_throughput_Gbps": compute_metrics["total_throughput_Gbps"],
                    "solution": compute_diagnostics["current_solution"],
                    "sinr_list": compute_diagnostics["sinr_list"],
                    }
        self.kpi_logger.log_metrics(
                        phase="environment",
                        algorithm="hybrid_marl",
                        metrics=metrics,
                        episode=self.episode_counter
                    )
    
    
    def get_last_info(self):
        """Return the last info dict from a step"""
        if hasattr(self, 'last_info'):
            print("Getting lastest info....")
            return self.last_info
        return None
    

    def _get_obs(self):
        """
        Get observation for each UE that includes:
        - Normalized SINR to each BS
        - PRB load fractions for each BS
        - BS utilization (rate/capacity ratio)
        - UE demand (normalized)
        - One-hot encoding of current association
        - Last-step throughput (bps/Hz) and global Jain fairness
        """
        # 1) PRB-based load fraction per BS        
        prb_fractions = get_prb_utilization(self.base_stations)  # returns array of shape (num_bs,) with values in [0,1]

        # 2) Rate-capacity utilization per BS        
        util_bps   = get_rate_utilization(self.base_stations)

        # 3) Compute global Jain's fairness index on PRB utilization
        global_jains = compute_jains_fairness(self.base_stations)  # scalar value for overall fairness

        obs = {}
        for ue in self.ues:
            # A) SINR vector normalized
            sinr_lin = calculate_sinrs(ue, self.base_stations).astype(np.float32)
            max_sinr = max(sinr_lin.max(), 1e-6)
            norm_sinr = sinr_lin / max_sinr

            # B) Normalized demand
            ue_demand = max(ue.demand, 1.0)
            norm_demand = np.array([min(ue.demand / ue_demand, 1.0)], dtype=np.float32)

            # C) One-hot association
            idx = ue.associated_bs if ue.associated_bs is not None else self.num_bs
            one_hot = np.eye(self.num_bs + 1, dtype=np.float32)[idx]

            # D) Last-step per-UE throughput (bps/Hz)
            last_throughput = np.array([compute_ue_throughput(ue)],dtype=np.float32)

            # E) Pack everything into one vector
            obs_vector = np.concatenate([
                norm_sinr,          # (num_bs,)
                prb_fractions,      # (num_bs,)
                # util_bps,           # (num_bs,)
                norm_demand,        # (1,)
                # one_hot,            # (num_bs+1,)
                last_throughput,    # (1,)
                # np.array([global_jains], dtype=np.float32)  # (1,)
            ], axis=0)

            obs[f"ue_{ue.id}"] = obs_vector
            # obs[ue.id] = obs_vector

        return obs

    
     
    
    def _update_system_metrics(self):
        """
        Refresh per-TTI system metrics:
        1) Updates each BS.load via calculate_load() (throughput or RB usage).
        2) Updates each UE.sinr as the average per-RB SINR over its allocated RBs.
        3) Append histories for PRBS masks and SINR.
        """
        # print("Updating System Metrics....")
        # 1) Recompute loads
        for bs in self.base_stations:
            bs.calculate_load()
            # print(f"For Update System Metrics to {bs.id}, Load :{bs.load}")
        # 2) Update UE SINRs based on actual RB allocations
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
            if ue.associated_bs is not None:
                bs = next(b for b in self.base_stations if b.id == ue.associated_bs)

                # Get per-RB SINRs
                sinr_rb = bs.snr_per_rb(ue)

                # Determine RBs allocated this TTI
                rb_list = bs.rb_allocation.get(ue.id, [])
                # print (f"RB List is:{rb_list}")
                if rb_list:
                    # Mean SINR over allocated RBs
                    ue.sinr = float(np.mean(sinr_rb[rb_list]))
                else:
                    ue.sinr = -np.inf
            else:
                ue.sinr = -np.inf
    
    def apply_solution(self, solution):
        """
        Apply a user↔BS association mapping and run the PRB-by-PRB PF scheduler
        on each BS for its set of UEs.
        """
        # --- 1) Normalize solution dict ---
        # print("Applying Solution to Environment......")
        if isinstance(solution, np.ndarray):
            sol_dict = {bs.id: [] for bs in self.base_stations}
            for ue_idx, bs_id in enumerate(solution.astype(int)):
                sol_dict[int(bs_id)].append(ue_idx)
            solution = sol_dict

        # --- 2) Validate BS and UE indices (as before) ---
        valid_bs_ids = {int(bs.id) for bs in self.base_stations}
        num_ues = len(self.ues)
        for bs_id, ue_list in solution.items():
            if int(bs_id) not in valid_bs_ids:
                raise ValueError(f"Invalid BS ID {bs_id}")
            for ue_id in ue_list:
                if ue_id < 0 or ue_id >= num_ues:
                    raise IndexError(f"UE index {ue_id} out of range")

        # --- 3) Clear out old allocs & associations ---
        for bs in self.base_stations:
            bs.rb_allocation = {ue_id: [] for ue_id in range(len(self.ues))}
            bs.allocated_resources.clear()
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
            ue.associated_bs = None

        # --- 4) Apply new associations ---
        for bs_id, ue_list in solution.items():
            bs = next(b for b in self.base_stations if b.id == int(bs_id))
            # Mark UEs as “in cell”
            for ue_id in ue_list:
                self.ues[ue_id].associated_bs = bs.id

        # --- 5) Run PF scheduler on each BS ---
        for bs in self.base_stations:
            # Only schedule the UEs currently associated
            ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues    
            active_ues = {ue.id: ue for ue in ue_iter if ue.associated_bs == bs.id}
            bs.ues = active_ues
            bs.allocate_prbs()    # PRB-by-PRB PF method
            
        # After apply_solution(solution) and allocate_prbs() have run:

        # Build a map ue_id → { bs_id, prbs: [...], sinrs_per_prb: [...] }
        alloc_details = {}

        # We need a fast lookup of UE objects by id
        ue_map = {ue.id: ue for ue in (self.ues.values() if isinstance(self.ues, dict) else self.ues)}

        for bs in self.base_stations:
            # bs.rb_allocation maps ue_id → list of PRB indices
            for ue_id, prb_list in bs.rb_allocation.items():
                if not prb_list:
                    continue

                # Compute the full SINR array for this UE
                ue = ue_map[ue_id]
                sinr_array = bs.snr_per_rb(ue)  # length = self.num_rbs

                # Extract only the SINRs on the allocated PRBs
                sinrs_on_prbs = sinr_array[prb_list]

                alloc_details[ue_id] = {
                    "bs_id": bs.id,
                    "prbs": prb_list,
                    "sinrs": sinrs_on_prbs.tolist()
                }

        # Inspect alloc_details, for example:
        # for ue_id, info in alloc_details.items():
        #     print(f"UE {ue_id} on BS {info['bs_id']},PRBs: {info['prbs']},SINRs (linear): {info['sinrs']}")
            

        # --- 6) Track load history & handovers ---
        for bs in self.base_stations:
            self.load_history[bs.id].append(bs.load)
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue in ue_iter:
        # for ue in self.ues:
            old = self.prev_associations[ue.id]
            new = ue.associated_bs
            if old is not None and new is not None and old != new:
                self.handover_counts[ue.id] += 1
            self.prev_associations[ue.id] = new
        self.step_count += 1

        # --- 7) Refresh per-UE SINR & BS loads ---
        self._update_system_metrics()

    def evaluate_detailed_solution(self, solution, alpha=0.1, beta=0.1):
        """
        Apply a candidate user-association solution, compute detailed performance metrics,
        then restore state. Returns metrics including average throughput in GB/s.
        """
        # # Update UE positions using RWP mobility model
        # for ue in self.ues:
        #     ue.update_position()
        
        # 1) Snapshot current state
        original = get_state_snapshot(self.ues, self.base_stations)

        # 2) Apply the proposed associations
        self.apply_solution(solution)

        # 3) Ensure loads and SINRs are up-to-date
        self.eval_recompute_network_state()

        solution_metrics = self.eval_compute_solution_metrics()     

        # 6) Restore original state
        set_state_snapshot(self, original)
        
        # 7) Return detailed report (throughput in GB/s)
        return solution_metrics
    
    def eval_get_ue_throughputs(env):
        throughputs = np.zeros(
            env.num_ue,
            dtype=np.float32
        )

        for bs in env.base_stations:
            for ue_id, rate in bs.allocated_resources.items():
                throughputs[ue_id] = rate

        return throughputs
    
    def eval_compute_solution_metrics(self):
        fitness     = self.calculate_global_reward()          # global reward
        # 4) Throughputs per UE from actual shared allocations (bits/sec)
        throughputs = self.eval_get_ue_throughputs()
        throughputs_mbps = throughputs / 1e6
        throughputs_gbps = throughputs / 1e9
        avg_throughput_Mbps = throughputs_mbps.mean()
        sum_throughputs=throughputs_gbps.sum()

        ue_iter = (
            self.ues.values()
            if isinstance(self.ues, dict)
            else self.ues
        )

        avg_sinr = np.mean(
            [ue.sinr for ue in ue_iter]
        )

        avg_sinr_db = 10 * np.log10(
            avg_sinr + 1e-12
        )

        throughput_variance = np.var(
            throughputs_gbps
        )

        fairness = (
            1.0 /
            (1.0 + throughput_variance)
        )

        load_var = np.var(
            [bs.load for bs in self.base_stations]
        )
        bs_loads    = [bs.load for bs in self.base_stations
                       ]
        # 7) Load‐distribution quantiles (Gbps)
        # concatenate all BS‐load histories (bps), convert to Gbps
        all_loads_bps = np.hstack([self.load_history[bs.id] for bs in self.base_stations])        
        all_loads_gbps = all_loads_bps / 1e9
        q10, q50, q90 = np.quantile(all_loads_gbps, [0.1, 0.5, 0.9])

        # 6) Handover rate
        total_handover_events = sum(self.handover_counts.values())
        ho_rate_per_step = total_handover_events / (self.num_ue * self.step_count)
        # # Debug: print a few sample UE stats in GB/s
        # for ue_id in throughputs.argsort()[-5:]:
        #     ue = self.ues[ue_id]            
        #     lin_snr = ue.sinr
        #     snr_db  = 10*np.log10(lin_snr + 1e-12)
        #     r_gbps = throughputs_Gbps[ue_id]
        #     print(f" UE {ue_id}: assoc→BS{ue.associated_bs}, "
        #         f"SINR={snr_db:.2f} dB, Rate={r_gbps:.3f} Gb/s") 
        return {
            "fitness": float(fitness),
            "average_sinr": float(avg_sinr_db),
            "average_throughput": float(avg_throughput_Mbps),
            "sum_throughput":float(sum_throughputs),
            "fairness": float(fairness),
            "load_variance": float(load_var),
            "bs_loads": bs_loads,
            "handover_rate": ho_rate_per_step,
            "load_quantiles_Gbps": {"10th": q10, "50th": q50, "90th": q90}

        }
    

    def eval_recompute_network_state(self):
        """Recompute all BS loads and UE SINRs based on current associations and allocations."""
        for bs in self.base_stations:
            bs.calculate_load()

        ue_iter = (
            self.ues.values()
            if isinstance(self.ues, dict)
            else self.ues
        )

        for ue in ue_iter:
            if ue.associated_bs is not None:
                sinr_vec = calculate_sinrs(
                    ue,
                    self.base_stations
                )
                ue.sinr = sinr_vec[ue.associated_bs]
            else:
                ue.sinr = -np.inf

        
# env = NetworkEnvironment({"num_ue": 3, "num_bs": 2})
# obs, _ = env.reset()
# print(obs["ue_0"].shape)  # Should be (2*2 + 1)=5

# actions = {"ue_0": 1, "ue_1": 0, "ue_2": 1}  # Each UE selects a BS index
# # next_obs, rewards, dones, _ = env.step(actions)
# next_obs, rewards, terminated, truncated, info = env.step(actions)
# print(next_obs, rewards, terminated, truncated, info)
# env1 = NetworkEnvironment({"num_ue": 10, "num_bs": 3})
# positions1 = [ue.position for ue in env1.ues]

# env2 = NetworkEnvironment({"num_ue": 10, "num_bs": 3})
# positions2 = [ue.position for ue in env2.ues]

# assert np.allclose(positions1, positions2)  # Should pass
# print(np.allclose(positions1, positions2))  # Should print: True
