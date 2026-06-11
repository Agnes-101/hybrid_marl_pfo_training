import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np

def get_state_snapshot(ues, base_stations) -> dict:
         # pick the UE iterable based on type
            
            ue_iter = (ues.values() if isinstance(ues, dict)
                else ues)
            return {
            "users": [{
                "id": ue.id,
                "position": ue.position.copy().tolist(),
                "waypoint": ue.waypoint.copy().tolist(),
                "speed": float(ue.speed),
                "pause_time": float(ue.pause_time),
                "demand": float(ue.demand),
                "associated_bs": ue.associated_bs,
                "sinr": float(ue.sinr),
                "ewma_dr": float(ue.ewma_dr)
            } for ue in ue_iter],# self.ues
            "base_stations": [{
                "id": bs.id,
                "allocated_resources": bs.allocated_resources.copy(),
                "rb_allocation": {ue_id: prbs.copy() for ue_id, prbs in bs.rb_allocation.items()},
                "load": float(bs.load),
                "capacity": float(bs.capacity),
                "reuse_color": bs.reuse_color,
                "frequency": float(bs.frequency)
            } for bs in base_stations]
            # # Optional: histories & counters
            # "load_history": {bs.id: hist.copy() for bs, hist in self.load_history.items()},
            # "handover_counts": self.handover_counts.copy(),
            # "prev_associations": self.prev_associations.copy(),
            # "step_count": self.step_count,
            # "current_step": self.current_step
        }

def set_state_snapshot(self, state: dict):
        # restore UEs
        ue_iter = self.ues.values() if isinstance(self.ues, dict) else self.ues
        for ue_state in state["users"]:
            ue = next(u for u in ue_iter if u.id == ue_state["id"])
            ue.position       = np.array(ue_state["position"], dtype=np.float32)
            ue.waypoint       = np.array(ue_state["waypoint"], dtype=np.float32)
            ue.speed          = ue_state["speed"]
            ue.pause_time     = ue_state["pause_time"]
            ue.demand         = ue_state["demand"]
            ue.associated_bs  = ue_state["associated_bs"]
            ue.sinr           = ue_state["sinr"]
            ue.ewma_dr        = ue_state["ewma_dr"]

        # restore BSs
        for bs_state in state["base_stations"]:
            bs = next(b for b in self.base_stations if b.id == bs_state["id"])
            bs.allocated_resources = bs_state["allocated_resources"].copy()
            # restore PRB maps too
            bs.rb_allocation = {int(uid): prbs.copy() 
                                for uid, prbs in bs_state["rb_allocation"].items()}
            bs.load        = bs_state["load"]
            bs.capacity    = bs_state["capacity"]
            bs.reuse_color = bs_state.get("reuse_color", bs.reuse_color)
            bs.frequency   = bs_state.get("frequency", bs.frequency)

        # # restore histories & counters
        # self.load_history      = {int(bs_id): hist.copy() 
        #                         for bs_id, hist in state.get("load_history", {}).items()}
        # self.handover_counts   = state.get("handover_counts", {}).copy()
        # self.prev_associations = state.get("prev_associations", {}).copy()
        # self.step_count        = state.get("step_count", self.step_count)
        # self.current_step      = state.get("current_step", self.current_step)   