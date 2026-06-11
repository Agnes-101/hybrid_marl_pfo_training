import sys
import os
# Add project root to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root) if project_root not in sys.path else None

import numpy as np

class BaseStation:
    """
    Base Station (BS) class representing a cellular base station in the environment.

    # The BS has key attributes such as:
    > BS ID,
    > BS Position on the 2D grid,   
    > Operating Frequency (Hz), Bandwidth (Hz), and Height (m),
    > Transmit Power (dBm), Antenna Gain (dBi), Beamforming Gain (dBi), and
    > Categorization whether 'Macro' or 'Small' through Reuse Color .

    # The BS can calculate the received power at a given UE position, 
        accounting for path loss (using a Close-In Reference Distance model) and environmental losses (rain, gas, cloud).  

    # The Received signal power is used to compute the SINR for each UE, 
       accounting for both thermal noise and inter-cell interference from neighboring base stations. 

    # The available system bandwidth is divided into Physical Resource Blocks (PRBs), 
        and achievable data rates are estimated using the Shannon capacity formula.
    
    # The BS can allocate Physical Resource Blocks (PRBs)
        to associated UEs based on their SINR and demand, using a demand-aware Proportional Fair (PF) scheduling algorithm
        that considers both instantaneous channel conditions and historical throughput (EWMA).

    
    """
    def __init__(self, id, position, frequency, bandwidth, height=50.0, reuse_color=None,tx_power_dbm=30.0,tx_gain_dbi=8.0,
                subcarrier_spacing=60e3, bf_gain_dbi=20.0, path_loss_n=2.0,path_loss_sigma=0.0,cre_bias=0.0):

        self.id = int(id)
        self.position = np.array(position, dtype=np.float32)
        self.frequency = float(frequency)    # Hz
        self.bandwidth = float(bandwidth)    # Hz
        self.height = float(height)          # m
        self.tx_power = float(tx_power_dbm)  # dBm
        self.tx_gain       = float(tx_gain_dbi)  # dBi
        self.bf_gain      = bf_gain_dbi   # beamforming gain
        self.allocated_resources = {}        # {ue_id: allocated_rate}
        self.load = 0.0                      # sum of allocated rates
        self.capacity = 0 # self.bandwidth * 0.8 # e.g. 80% of bandwidth in Mbps
        self.reuse_color   = reuse_color         # e.g. "A", "B", or "C"
        self.path_loss_n     = float(path_loss_n)
        self.path_loss_sigma = float(path_loss_sigma)
        self.cre_bias        = float(cre_bias)
        # Resource block parameters
        self.rb_bandwidth = 12*subcarrier_spacing # Hz (typical 5G/6G subcarrier spacing × 12)
        self.num_rbs = int(self.bandwidth / self.rb_bandwidth)  # Number of available RBs
        # print(f"BS with reuse Colour: {reuse_color},RB Bandwidth: {self.rb_bandwidth}, No. Num_rbs:{self.num_rbs}")
        self.seed = 42
        if self.seed is not None:
            np.random.seed(self.seed+self.id)
        
        self.rb_allocation = {}  # Dictionary mapping UE IDs to allocated RBs
        self.rb_sinr = np.zeros(self.num_rbs)  # SINR per RB (for interference modeling)
        
    def calculate_load(self):
        # total number of RBs in use:
        used_rbs = sum(len(rbs) for rbs in self.rb_allocation.values())
        # if you want absolute count:
        self.load = used_rbs

    # def path_loss(self, distance, ue_height=1.5):
    #     # Okumura-Hata suburban model
    #     f = self.frequency
    #     hb = self.height
    #     hu = ue_height
    #     ch = 0.8 + (1.1 * np.log10(f) - 0.7) * hu - 1.56 * np.log10(f)
    #     const1 = 69.55 + 26.16 * np.log10(f) - 13.82 * np.log10(hb) - ch
    #     const2 = 44.9 - 6.55 * np.log10(hb)
    #     return const1 + const2 * np.log10(distance + 1e-9)
    
    def path_loss(self, distance, d0=1.0):
        """
        Close-In Reference Distance Path-Loss Model.
        distance: link distance in meters
        d0: reference distance (m), typically 1.0
        n: path-loss exponent (environment-specific)
        sigma: shadow-fading std. dev. (dB)
        """
        c = 3e8  # speed of light (m/s)

        # 1) Free‐space loss at d0 (usually 1 m):       
        fspl_d0 = 20*np.log10(4*np.pi*d0*self.frequency/c)

        # 2) Distance‐dependent term        
        pl_mean = fspl_d0 + 10*self.path_loss_n*np.log10(distance/d0)

        # 3) Add shadow fading (dB)        
        shadow   = np.random.randn()*self.path_loss_sigma
        return pl_mean + shadow
    

    def noise_mW(self):
        # Thermal noise: -174 dBm/Hz + 10*log10(BW_Hz)
        noise_dbm = -174 + 10 * np.log10(self.bandwidth )
        return 10 ** (noise_dbm / 10)
    
    
    def _calculate_local_interference(self, neighbor_dist=None):
        # Set default neighbor distance by tier
        if neighbor_dist is None:
            neighbor_dist = 1000.0 if self.reuse_color=="Macro" else 100.0

        interference = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue

            d = np.linalg.norm(self.position - other.position)
            if d > neighbor_dist:
                continue

            # other.received_power_mW already includes path-loss
            interference += other.received_power_mW(self.position)

        return interference


    
    # def received_power_mW(self, ue_pos, ue_rx_gain=None):# , ue_bf_gain_dbi=0.0
    #     d = np.linalg.norm(self.position - ue_pos)
    #     # The code is calculating the path loss using a function `path_loss` with the distance `d` as
    #     # an input parameter and storing the result in the variable `L`.
    #     L  = self.path_loss(d)
    #     G_tx = self.tx_gain + self.bf_gain      # total transmit gain
    #     G_rx = ue_rx_gain if ue_rx_gain is not None else 0.0 # (ue_rx_gain or 0.0) + ue_bf_gain_dbi  # UE may also beam-form
    #     p_rx_dbm = self.tx_power + G_tx + G_rx - L
    #     return 10**(p_rx_dbm/10)
    def received_power_mW(self, ue_position, ue_rx_gain=None):
        """Calculate received power with atmospheric/environmental losses"""
        # 1. Basic free space path loss
        distance = np.linalg.norm(ue_position - self.position)
        lambda_ = 3e8 / (self.frequency * 1e6)  # Wavelength in meters
        
        # Free space path loss (dB)
        fspl = 20 * np.log10(distance) + 20 * np.log10(self.frequency) + 20 * np.log10(4 * np.pi / 3e8)
        
        # 2. Atmospheric attenuation components (ITU-R models)
        def rain_attenuation(freq_ghz, rain_rate, distance_km):
            """ITU-R P.838-8 specific attenuation due to rain"""
            k = 0.0335 * (freq_ghz**0.9)  # Coefficients for 10-100 GHz
            alpha = 1 - 0.02 * (freq_ghz - 10)
            return k * (rain_rate**alpha) * distance_km

        def gaseous_attenuation(freq_ghz, temp=15, humidity=60):
            """ITU-R P.676-13 oxygen/water vapor absorption"""
            # Simplified model for 1-60 GHz
            if freq_ghz < 15:
                return 0.05 * freq_ghz  # dB/km
            elif 15 <= freq_ghz < 60:
                return 0.1 * (freq_ghz - 14) + 0.75  # dB/km
            else:
                return 3.0  # dB/km (approximate for 60+ GHz)

        # 3. Environmental parameters (customize these)
        rain_rate = 10  # mm/h - moderate rain
        humidity = 60    # %
        temperature = 25  # °C
        
        # Calculate additional losses
        distance_km = distance / 1000
        freq_ghz = self.frequency / 1e3
        
        L_rain = rain_attenuation(freq_ghz, rain_rate, distance_km)
        L_gas = gaseous_attenuation(freq_ghz, temperature, humidity) * distance_km
        L_cloud = 0.1 * distance_km  # Simple cloud attenuation
        
        # 4. Antenna and system losses
        L_polarization = 1  # dB - polarization mismatch
        L_pointing = 0.5    # dB - beam misalignment
        
        # 5. Total additional losses
        total_add_loss = L_rain + L_gas + L_cloud + L_polarization + L_pointing
        
        # 6. Final received power calculation
        Prx_dBm = (self.tx_power
                + self.tx_gain 
                - fspl 
                - total_add_loss)
        # Macro-specific fixed attenuation (IN dB DOMAIN)
        if self.reuse_color == "Macro":
            fixed_macro_loss = 50  # dB
            Prx_dBm -= fixed_macro_loss  # Proper dB subtraction
        else:
            fixed_macro_loss = 5  # dB
            Prx_dBm -= fixed_macro_loss  # Proper dB subtraction
            
        return 10 ** (Prx_dBm / 10)  # Convert dBm to mW
    
    def snr_linear(self, ue, cross_tier=True, max_sinr_db=40.0):
        """
        Compute the linear SINR for this BS → ue link, including
        Tx/Rx gains, co-channel (or cross-tier) interference, and thermal noise.
        
        cross_tier: if True, include interference from all BSs; 
                    if False, only from same reuse_color.
        max_sinr_db: cap SINR at this dB value (e.g. 50 dB).
        """

        # 1) desired signal power (mW)
        signal_mW = self.received_power_mW(ue.position, ue.rx_gain)

        # 2) build interference sum
        interference_mW = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue

            if not cross_tier:
                # only co-channel
                if other.reuse_color != self.reuse_color:
                    continue

            # include this BS’s contribution
            interference_mW += other.received_power_mW(ue.position, ue.rx_gain)

        # 3) noise power (mW)
        noise_mW = self.noise_mW()

        # 4) raw linear SINR
        lin_snr = signal_mW / (interference_mW + noise_mW + 1e-12)

        # 5) apply cap
        snr_cap_linear = 10 ** (max_sinr_db / 10)
        return min(lin_snr, snr_cap_linear)

    
    def allocate_prbs(self):
        """
        Improved version of the PRB allocation method that ensures fairer distribution
        and resolves bugs in the original implementation.
        """
        # Handle empty case
        if not self.ues:
            self.rb_allocation = {}
            self.allocated_resources = {}
            self.calculate_load()
            return
            
        # Determine UE IDs properly
        if isinstance(self.ues, dict):
            ue_ids = list(self.ues.keys())
        else:
            # self.ues is a list of UE objects
            ue_ids = [ue.id for ue in self.ues]

        self.rb_allocation = {ue_id: [] for ue_id in ue_ids}
        delivered_bits = {ue_id: 0.0 for ue_id in ue_ids}

        # Fast path for single UE case
        if len(self.ues) == 1:
            ue_id = next(iter(self.ues))
            ue = self.ues[ue_id]
            max_bits = ue.demand * 1e6  # demand in bits/sec

            # Tentatively give it all PRBs
            prb_list = list(range(self.num_rbs))
            sinr_array = self.snr_per_rb(ue)
            rate_array = self.rb_bandwidth * np.log2(1 + sinr_array)
            total_bits = rate_array.sum()

            # Cap at demand
            allocated_bits = min(total_bits, max_bits)
            # Figure out how many PRBs to use to stay under demand
            if allocated_bits < total_bits:
                # allocate highest‐rate PRBs until sum(rate) ≥ max_bits
                order = np.argsort(-rate_array)
                cum = 0.0
                prb_list = []
                for prb in order:
                    cum += rate_array[prb]
                    prb_list.append(prb)
                    if cum >= max_bits:
                        break
            # Store the allocation
            self.rb_allocation[ue_id] = prb_list
            delivered_bits[ue_id] = min(max_bits, sum(rate_array[prb] for prb in prb_list))
            # skip the rest of the algorithm
            self.allocated_resources = {ue_id: delivered_bits[ue_id]}
            self.calculate_load()
            return
        
        # OPTIMIZATION 1: Efficient Precomputation with Caching
        # --------------------------------------------------
        
        # Precompute SINR arrays, rates, and best PRB ordering for each UE
        sinr_map = {}               # Store raw SINR values
        sinr_db_map = {}            # Store SINR in dB for clustering
        rate_map = {}               # Store achievable rates per PRB
        sorted_prb_indices = {}     # Store pre-sorted PRB indices (best first)
        avg_sinr_db = {}            # Store average SINR in dB per UE
        
        for ue_id, ue in self.ues.items():
            # Get SINR for all PRBs at once (single calculation)
            sinr_array = self.snr_per_rb(ue)
            sinr_map[ue_id] = sinr_array
            
            # Calculate SINR in dB for clustering
            sinr_db_array = 10 * np.log10(sinr_array + 1e-10)  # Avoid log of zero
            sinr_db_map[ue_id] = sinr_db_array
            
            # Precalculate rates for all PRBs
            rate_map[ue_id] = self.rb_bandwidth * np.log2(1 + sinr_array)
            
            # Calculate average SINR for clustering
            avg_sinr_db[ue_id] = float(np.mean(sinr_db_array))
            
            # Precompute sorted PRB indices (best to worst)
            sorted_prb_indices[ue_id] = np.argsort(-sinr_array)
        
        # IMPROVED: Better UE Clustering with More Flexible Boundaries
        # --------------------------------------------------
        
        # Sort UEs by average SINR
        sorted_ues = sorted(avg_sinr_db.items(), key=lambda x: x[1])
        
        # Create more adaptable SINR clusters with dynamic boundaries
        # Use percentile-based approach instead of fixed dB difference
        num_clusters = min(5, len(sorted_ues))  # Limit max clusters based on total UEs
        cluster_size = max(1, len(sorted_ues) // num_clusters)
        
        sinr_clusters = []
        for i in range(0, len(sorted_ues), cluster_size):
            cluster = [ue_id for ue_id, _ in sorted_ues[i:i+cluster_size]]
            if cluster:  # Only add non-empty clusters
                sinr_clusters.append(cluster)
        
        # OPTIMIZATION 3: Efficient PRB Tracking
        # --------------------------------------------------
        
        # Use boolean array for PRB availability
        available_prbs = np.ones(self.num_rbs, dtype=bool)
        
        # IMPROVED: Better Initial Allocation Strategy
        # --------------------------------------------------
        
        # First ensure every UE gets at least one PRB to avoid starvation
        # Prioritize UEs with poorer conditions for fairer initial allocation
        
        # Start with UEs in clusters from worst to best SINR
        for cluster in sinr_clusters:
            for ue_id in cluster:
                # Skip if already allocated in a previous pass
                if self.rb_allocation[ue_id]:
                    continue
                    
                # Find best available PRB for this UE
                for prb_idx in sorted_prb_indices[ue_id]:
                    if available_prbs[prb_idx]:
                        # Allocate this PRB
                        self.rb_allocation[ue_id].append(prb_idx)
                        available_prbs[prb_idx] = False
                        delivered_bits[ue_id] += rate_map[ue_id][prb_idx]
                        break  # One PRB per UE in this phase
        
        # IMPROVED: Proportional Fairness Allocation with Demand Awareness
        # --------------------------------------------------
        
        # Calculate normalized demand for each UE (for demand-aware allocation)
        max_demand = max(ue.demand for _, ue in self.ues.items())
        normalized_demand = {ue_id: self.ues[ue_id].demand / max_demand 
                            for ue_id in self.ues}
        
        # Calculate how many PRBs are still available
        remaining_prbs_count = np.sum(available_prbs)
        
        # If PRBs remain, allocate them using improved PF with demand awareness
        if remaining_prbs_count > 0:
            # Get remaining PRB indices
            remaining_prb_indices = np.where(available_prbs)[0]
            
            # Allocate remaining PRBs
            for prb in remaining_prb_indices:
                best_metric = -float('inf')
                best_ue = None
                
                for ue_id in self.ues:
                    # Skip if demand is already satisfied
                    current_rate = delivered_bits[ue_id]
                    max_demand_bits = self.ues[ue_id].demand * 1e6
                    
                    if current_rate >= max_demand_bits:
                        continue
                    
                    # Avoid division by zero with a safe minimum EWMA
                    ewma = max(self.ues[ue_id].ewma_dr, 1e-6)
                    
                    # Improved metric calculation:
                    # - Higher weight for UEs far from their demand
                    # - Consider both instantaneous rate and long-term fairness
                    demand_satisfaction = current_rate / max_demand_bits if max_demand_bits > 0 else 1.0
                    demand_weight = 1.0 / (0.1 + 0.9 * demand_satisfaction)  # Higher weight for unsatisfied demand
                    
                    # Combine rate, fairness, and demand factors
                    metric = (rate_map[ue_id][prb] / ewma) * demand_weight
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_ue = ue_id
                
                # Allocate PRB to best UE if found
                if best_ue is not None:
                    self.rb_allocation[best_ue].append(prb)
                    available_prbs[prb] = False
                    delivered_bits[best_ue] += rate_map[best_ue][prb]
        
        # IMPROVED: Final Rebalancing for Zero-Allocation UEs
        # --------------------------------------------------
        
        # Check for UEs with zero PRBs - more aggressive redistribution
        zero_prb_ues = [ue_id for ue_id, prbs in self.rb_allocation.items() if not prbs]
        
        if zero_prb_ues:
            # Find UEs with multiple PRBs as potential donors
            ue_prb_counts = [(ue_id, len(prbs), delivered_bits[ue_id] / (self.ues[ue_id].demand * 1e6)) 
                            for ue_id, prbs in self.rb_allocation.items() if len(prbs) > 1]
            
            # Sort donors by satisfaction ratio (most satisfied first)
            ue_prb_counts.sort(key=lambda x: x[2], reverse=True)
            
            # Try to help each zero-PRB UE
            for zero_ue in zero_prb_ues:
                # Find best donor with multiple PRBs
                for donor_idx, (donor_ue, prb_count, _) in enumerate(ue_prb_counts):
                    if prb_count <= 1:  # Don't take from UEs with only 1 PRB
                        continue
                    
                    # Take the worst PRB from donor (least impact to donor)
                    donor_prbs = self.rb_allocation[donor_ue]
                    donor_rates = [rate_map[donor_ue][prb] for prb in donor_prbs]
                    worst_idx = np.argmin(donor_rates)
                    prb = donor_prbs[worst_idx]
                    
                    # Only donate if it helps zero_ue more than it hurts donor_ue
                    if rate_map[zero_ue][prb] > rate_map[donor_ue][prb] * 0.8:  # 20% efficiency threshold
                        # Transfer the PRB
                        donor_prbs.pop(worst_idx)
                        self.rb_allocation[zero_ue].append(prb)
                        
                        # Update delivered bits
                        delivered_bits[zero_ue] += rate_map[zero_ue][prb]
                        delivered_bits[donor_ue] -= rate_map[donor_ue][prb]
                        
                        # Update donor's PRB count
                        ue_prb_counts[donor_idx] = (donor_ue, prb_count - 1, 
                                                delivered_bits[donor_ue] / (self.ues[donor_ue].demand * 1e6))
                        ue_prb_counts.sort(key=lambda x: x[2], reverse=True)
                        break
        
        
        # Final EWMA Update and Resource Calculation
        # --------------------------------------------------
        
        # Update EWMA throughput for each UE
        for ue_id, ue in self.ues.items():
            ue.update_ewma(delivered_bits[ue_id])
            if ue.ewma_dr < 1e-6:  # Set a minimal EWMA
                ue.ewma_dr = 1e-6
        
        # Calculate allocated resources in bps
        self.allocated_resources = {}
        for ue_id, prbs in self.rb_allocation.items():
            if prbs:  # Only calculate for UEs with allocated PRBs
                self.allocated_resources[ue_id] = sum(rate_map[ue_id][prb] for prb in prbs)
            else:
                self.allocated_resources[ue_id] = 0.0
        
        # # Display allocation results
        # for ue_id, ue in self.ues.items():
        #     alloc_mbps = self.allocated_resources.get(ue_id, 0.0) / 1e6
        #     print(f" BS {self.reuse_color},UE {ue_id}: allocated={alloc_mbps:.2f} Mbps, demand={ue.demand:.2f} Mbps")

        # Calculate system load
        self.calculate_load()
        

    def calculate_capacity_rb_based(self, sample_points=500, overhead_factor=0.8):
        """
    Estimates BS rb capacity using Monte Carlo sampling of user locations.

    Capacity is computed from the average spectral efficiency obtained
    from SINR measurements at randomly sampled points within the cell
    coverage area. 
    
    The result is scaled by the number of available
    resource blocks and an overhead factor to account for losses.
    """
        total_se = 0.0
        
        # Debug for macro
        is_macro = self.reuse_color == "Macro"
        
        # 1) Determine a coverage radius - FIXED APPROACH
        # For macro: standard coverage radius
        # For small cells: either use distances between cells
        if is_macro:
            cell_radius = 500.0  # More realistic macro coverage radius
        else:
            # For small cells, use distance to nearest neighbor or default
            same_tier_bs = [bs for bs in self.base_stations 
                        if bs.id != self.id and bs.reuse_color == self.reuse_color]
            
            if same_tier_bs:
                cell_radius = min(
                    np.linalg.norm(bs.position - self.position)
                    for bs in same_tier_bs ) / 2.0  # Half the distance to nearest same-tier BS
            else:
                cell_radius = 100.0  # Default small cell radius
        
        # 2) Cap instantaneous SINR at 30 dB (≈1000 linear)
        max_sinr_db = 30.0
        sinr_cap = 10 ** (max_sinr_db / 10)
        
        # Track average SINR for debugging
        sinr_samples = []
        
        # 3) Uniform‐disk sampling
        for _ in range(sample_points):
            u = np.random.rand()          # uniform [0,1)
            r = np.sqrt(u) * cell_radius  # uniform area
            theta = np.random.rand() * 2 * np.pi
            dx, dy = r * np.cos(theta), r * np.sin(theta)
            sample_point = self.position + np.array([dx, dy], dtype=np.float32)
            
            # 4) Compute desired & cross‐tier interference + noise
            prx = self.received_power_mW(sample_point)
            
            # Consider different interference models for macro vs small cells
            if is_macro:
                # For macro, small cells generally operate in different frequency bands
                # so they cause minimal interference
                interf = sum(
                    other.received_power_mW(sample_point) * 0.01  # Reduced cross-tier interference
                    for other in self.base_stations
                    if other.id != self.id and other.reuse_color != "Macro"  # Only from other tiers
                )
            else:
                # Small cells see interference from same color cells and reduced from macro
                interf = sum(
                    other.received_power_mW(sample_point) * (1.0 if other.reuse_color == self.reuse_color else 0.1)
                    for other in self.base_stations
                    if other.id != self.id
                )
                
            noise = self.noise_mW()
            
            sinr = min(prx / (interf + noise + 1e-12), sinr_cap)
            sinr_samples.append(sinr)
            total_se += np.log2(1 + sinr)
        
        # 5) From average spectral efficiency to capacity
        avg_se = total_se / sample_points       # bits/s/Hz
        rb_cap = self.rb_bandwidth * avg_se     # bits/s per RB
        total_bps = rb_cap * self.num_rbs * overhead_factor
        
        # Safety minimum capacities based on technology
        min_capacity = 100.0 if is_macro else 50.0
        
        # Mbps, with a guardrail at 10 Gbps and minimum capacity
        self.capacity = max(min(total_bps / 1e6, 1e4), min_capacity)
        
        # # Additional debug for macro
        # if self.id == 0:
        #     avg_sinr = sum(sinr_samples) / len(sinr_samples)
        #     print(f"Cell radius: {cell_radius:.1f} m")
        #     print(f"Avg SINR: {10*np.log10(avg_sinr):.2f} dB")
        #     print(f"Avg SE: {avg_se:.4f} bits/s/Hz")
        #     print(f"RB capacity: {rb_cap/1e6:.4f} Mbps")
        #     print(f"Total theoretical: {total_bps/1e6:.4f} Mbps")
        #     print(f"Final capacity: {self.capacity:.4f} Mbps")
        #     print("-----------------------------------")
        
        return self.capacity
    
    def snr_per_rb(self, ue):
        """
        Computes per-resource-block SINR for a UE.

        Applies frequency-selective fading across RBs(Rayleigh), computes received signal
        power per RB, and returns SINR values accounting for interference and noise.
        """
        sinr_rb = np.empty(self.num_rbs, dtype=np.float32)
        noise_rb = self.noise_mW()
        
        # Calculate base received power
        base_prx = self.received_power_mW(ue.position, ue.rx_gain)
        
        # Generate frequency-selective fading per RB using a correlated Rayleigh fading model
        # with correlation across adjacent RBs
        coherence_rbs = min(20, self.num_rbs // 50)  # Coherence bandwidth in RBs
        num_independent_fades = max(1, self.num_rbs // coherence_rbs)
        
        # Generate independent fades
        independent_fades = np.random.rayleigh(scale=1.0, size=num_independent_fades)
        
        # Interpolate to get per-RB fading
        fading = np.interp(
            np.linspace(0, num_independent_fades-1, self.num_rbs),
            np.arange(num_independent_fades),
            independent_fades
        )
        
        # Normalize fading to maintain average power
        fading = fading / np.mean(fading)
        
        # Apply fading to received power per RB
        prx_per_rb = base_prx * fading
        
        # Calculate interference from other BSs (same-tier and cross-tier with different weights)
        interf = 0.0
        for other in self.base_stations:
            if other.id == self.id or other.reuse_color != self.reuse_color:
                continue
            interf += other.received_power_mW(ue.position, ue.rx_gain)
        
        # Calculate SINR per RB
        for rb in range(self.num_rbs):
            sinr_rb[rb] = prx_per_rb[rb] / (interf + noise_rb + 1e-12)
        
        return sinr_rb
    