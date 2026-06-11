import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np

class UE:
    """
    User Equipment (UE) class representing a mobile user in the environment.
    
    # Here we model a user with key attributes such as 
    > User ID,
    > User RandomizedPosition on the 2D grid, 
    > Randomized Resource allocation Demand in terms of resource blocks (RBs) or data rate (Mbps), and
    > Mobility characteristics such as the minimum and maximum speed, and pause times.

    # The UE's movement is modelled according to the Random Waypoint (RWP) model, 
        which includes random pauses and variable speeds. 
    Ref 1 : https://www.mathworks.com/help/wireless-network/ug/overview-of-mobility-models.html#mw_ebd4d81d-bdd3-4889-bce6-5666bf88cffa
    Ref 2 : https://www.sciencedirect.com/topics/engineering/random-waypoint-model
        
    # Additionally, the UE can calculate its required Resource Blocks (RBs) 
        based on its demand and the signal quality from a given Base Station (BS).

    # Resource allocation is performed using a Proportional Fair (PF) scheduler,
        which doesn't always serve the user with the best channel conditions, 
        but rather allocates resources to users based on a ratio of their current achievable data rate(the UE's demand)
        to their historical average data rate(using an Exponentially Weighted Moving Average, EWMA)
    Ref : https://en.wikipedia.org/wiki/Proportional-fair_scheduling

    # Historical throughput is maintained using an Exponentially Weighted Moving Average (EWMA), 
        where recent throughput measurements are given greater weight than older observations 
    
    """
    def __init__(self, id, position, demand,
                v_min=0.1, v_max=0.6,
                pause_min=1.0, pause_max=5.0,rx_gain_dbi=0.0,
                ewma_alpha=0.7):
        self.id          = int(id)
        self.position    = np.array(position, dtype=np.float32)  
        self.demand      = float(demand)       # Mbps
        # RWP fields:
        self.v_min       = v_min               # min speed (m/s)
        self.v_max       = v_max               # max speed (m/s)
        self.pause_min   = pause_min           # min pause (s)
        self.pause_max   = pause_max           # max pause (s)
        self.waypoint    = self._draw_waypoint()
        self.speed       = np.random.uniform(v_min, v_max)
        self.pause_time  = 0.0                 # start “moving”
        self.seed = 42

        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Scheduling fields:    
        self.associated_bs = None
        self.sinr          = -np.inf
        self.ewma_dr       = 1e6
        self.ewma_alpha    = ewma_alpha
        self.rx_gain = float(rx_gain_dbi)  # dBi


    def _draw_waypoint(self):
        # uniformly anywhere in the 100×100 area
        return np.random.uniform(0, 100, size=2).astype(np.float32)

    def update_position(self, delta_time=0.08):
        """RWP: if paused, count down; else move toward waypoint."""
        if self.pause_time > 0:
            # still in pause
            self.pause_time -= delta_time
            return

        # vector and distance to waypoint
        direction = self.waypoint - self.position
        dist      = np.linalg.norm(direction)
        # how far we’d travel this step
        travel    = self.speed * delta_time

        if travel >= dist:
            # reached (or overshot) the waypoint
            self.position   = self.waypoint.copy()
            # draw a new pause interval
            self.pause_time = np.random.uniform(self.pause_min, self.pause_max)
            # pick next waypoint & speed
            self.waypoint   = self._draw_waypoint()
            self.speed      = np.random.uniform(self.v_min, self.v_max)
        else:
            # move fractionally toward the waypoint
            self.position += (direction / dist) * travel

    def update_ewma(self, measured_dr):
        self.ewma_dr = self.ewma_alpha * self.ewma_dr + (1 - self.ewma_alpha) * measured_dr
    
    # def update_ewma(self, allocated_rb):
    #     # EWMA of RBs per TTI
    #     self.ewma_rb = self.ewma_alpha * self.ewma_rb + (1 - self.ewma_alpha) * allocated_rb
                     
    def get_required_rbs(self, bs):
        """Calculate how many RBs this UE needs from this BS"""
        sinr = self._calculate_sinrs(bs, 0)  # Using RB 0 as reference
        spectral_efficiency = np.log2(1 + sinr)  # bits/s/Hz
        rb_capacity = bs.rb_bandwidth * 1e6 * spectral_efficiency  # bits/s per RB
        
        # RBs needed to satisfy demand (bps)
        return int(np.ceil((self.demand * 1e6) / rb_capacity))