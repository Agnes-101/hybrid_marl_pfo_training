import numpy as np
import torch

### ----BS Resource Metrics---- ###
def get_prb_utilization(base_stations):
    bs_prb_loads = np.array(
        [bs.load for bs in base_stations],
        dtype=np.float32
    )

    bs_num_rbs = np.array(
        [max(bs.num_rbs, 1) for bs in base_stations],
        dtype=np.float32
    )

    return bs_prb_loads / bs_num_rbs
def get_rate_utilization(base_stations):
    """
    BS utilization based on allocated rate / BS capacity.
    
    Returns
    -------
    np.ndarray
        Utilization per BS.
    """
    rate_loads = np.array(
        [
            sum(bs.allocated_resources.values())
            for bs in base_stations
        ],
        dtype=np.float32,
    )

    capacities = np.array(
        [
            max(bs.capacity * 1e6, 1.0)
            for bs in base_stations
        ],
        dtype=np.float32,
    )

    return np.clip(
        rate_loads / capacities,
        0.0,
        10.0,
    )

#### ----UE Performance Metrics---- ####
def calculate_sinrs(ue, base_stations, debug=False):
    """
        Return an array of linear SINRs from every BS → UE link.
        If debug=True, also print the first few link budgets in dB.
    """
    sinrs = []
    for bs in base_stations:
            # 1) desired signal power (mW)
        prx = bs.received_power_mW(ue.position, ue.rx_gain)

            # 2) co-channel interference (mW)
        interf = sum(
                other.received_power_mW(ue.position, ue.rx_gain)
                for other in base_stations
                if (other.id != bs.id) and (other.reuse_color == bs.reuse_color)
            )

            # 3) noise power (mW)
        noise = bs.noise_mW()

            # 4) linear SINR
        lin_snr = prx / (interf + noise + 1e-12)

            # Optional debug print in dB
        if debug and ue.id < 3:
            snr_db = 10 * np.log10(lin_snr)
            print(f" UE {ue.id} → BS {bs.id}: "
                f"S={prx:.3e} mW, I={interf:.3e} mW, N={noise:.3e} mW, "
                f"SINR={snr_db:.2f} dB")

        sinrs.append(lin_snr)

    return np.array(sinrs)


    
def _update_sinrs(self):
    for ue in self.ues:
        if ue.associated_bs is not None:
                sinrs = self._calculate_sinrs(ue)
                ue.sinr = sinrs[ue.associated_bs]
        else:
                ue.sinr = -np.inf

def _calculate_local_interference(self, neighbor_dist=100.0):
        interference = 0.0
        for other in self.base_stations:
            if other.id == self.id:
                continue
            # only count BSs on the same reuse_color
            if other.reuse_color != self.reuse_color:
                continue
            d = np.linalg.norm(self.position - other.position)
            if d < neighbor_dist:
                prx_int = other.received_power_mW(self.position)
                interference += prx_int
        return interference


# Add these to your NetworkEnvironment class
def calculate_jains_fairness(self):
    throughputs = [ue.throughput for ue in self.ues]
    return (sum(throughputs) ** 2) / (len(throughputs) * sum(t**2 for t in throughputs))

    # @property
def throughput(self):
    return torch.log2(1 + 10**(self.sinr/10)).item()

def compute_ue_throughput( ue):
    if ue.associated_bs is None:
        return 0.0

    sinr_db = min(ue.sinr, 100.0)

    return float(
        np.log2(1 + 10 ** (sinr_db / 10))
    )

def compute_jains_fairness(base_stations):
    util = get_prb_utilization(base_stations)

    if util.sum() > 0:
        return float((util.sum() ** 2)
        /
        (len(util) * (util ** 2).sum() + 1e-9)
    )
    else:
        return 0.0