"""
Unified SINR, Capacity, and Coverage calculations for 
Terrestrial Networks (TN) and Non-Terrestrial Networks (NTN).

References:
- TN: 3GPP TR 38.901 (Pathloss and Scenarios)
- NTN: 3GPP TR 38.821 (Satellite link budgets)
"""

import math
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 1. Constants & Enums
# ──────────────────────────────────────────────────────────────────────────────

C_M_S   = 299_792_458.0   # Speed of light [m/s]
K_B     = 1.380649e-23    # Boltzmann constant [J/K]
T_SYS_K = 290.0           # Reference system noise temperature [K]
K_DB    = -228.6          # Boltzmann constant [dBW/K/Hz]

class DeploymentScenario(Enum):
    UMA    = "Urban Macro"
    UMI    = "Urban Micro"
    RMA    = "Rural Macro"
    INH    = "Indoor Hotspot"
    INF_SH = "Indoor Factory (Sparse High)"

DEFAULT_H_BS = {
    DeploymentScenario.UMA: 25.0,
    DeploymentScenario.UMI: 10.0,
    DeploymentScenario.RMA: 35.0,
    DeploymentScenario.INH: 3.0,
    DeploymentScenario.INF_SH: 8.0
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Internal Helpers & 3GPP Pathloss Engine
# ──────────────────────────────────────────────────────────────────────────────

def _fspl_db_km_ghz(distance_km: float, freq_ghz: float) -> float:
    """FSPL [dB] using the telecom shorthand constant 92.45."""
    return (20.0 * math.log10(max(distance_km, 1e-3))
            + 20.0 * math.log10(max(freq_ghz, 1e-6))
            + 92.45)

def _uma_los_pathloss(d_2d: float, d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> float:
    h_e = 1.0 # Effective environment height
    d_bp = 4.0 * (h_bs - h_e) * (h_ut - h_e) * (fc_ghz * 1e9) / C_M_S
    if d_2d <= d_bp:
        return 28.0 + 22.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
    else:
        return (28.0 + 40.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz) 
                - 9.0 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))

def _umi_los_pathloss(d_2d: float, d_3d: float, fc_ghz: float, h_bs: float, h_ut: float) -> float:
    d_bp = 4.0 * h_bs * h_ut * (fc_ghz * 1e9) / C_M_S
    if d_2d <= d_bp:
        return 32.4 + 21.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz)
    else:
        return (32.4 + 40.0 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz) 
                - 9.5 * math.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))

def pathloss_3gpp_nlos(
    scenario: DeploymentScenario,
    distance_2d_m: float,
    carrier_freq_hz: float,
    ue_height_m: float = 1.5,
    bs_height_m: Optional[float] = None,
) -> float:
    """Computes deterministic NLOS path loss for any 3GPP Scenario."""
    fc_ghz = carrier_freq_hz / 1e9
    min_dist = 1.0 if scenario in [DeploymentScenario.INH, DeploymentScenario.INF_SH] else 10.0
    d_2d = max(distance_2d_m, min_dist)
    
    if bs_height_m is None:
        bs_height_m = DEFAULT_H_BS[scenario]

    d_3d = math.sqrt(d_2d ** 2 + (bs_height_m - ue_height_m) ** 2)

    if scenario == DeploymentScenario.UMA:
        pl_nlos = 13.54 + 39.08 * math.log10(d_3d) + 20.0 * math.log10(fc_ghz) - 0.6 * (ue_height_m - 1.5)
        pl_los = _uma_los_pathloss(d_2d, d_3d, fc_ghz, bs_height_m, ue_height_m)
        return max(pl_los, pl_nlos)
        
    elif scenario == DeploymentScenario.UMI:
        pl_nlos = 35.3 * math.log10(d_3d) + 22.4 + 21.3 * math.log10(fc_ghz) - 0.3 * (ue_height_m - 1.5)
        pl_los = _umi_los_pathloss(d_2d, d_3d, fc_ghz, bs_height_m, ue_height_m)
        return max(pl_los, pl_nlos)
        
    # Default to UMa if an unsupported scenario is passed
    return pathloss_3gpp_nlos(DeploymentScenario.UMA, distance_2d_m, carrier_freq_hz, ue_height_m, bs_height_m)


# ──────────────────────────────────────────────────────────────────────────────
# 3. TN SINR & Capacity (5G NR)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_tn_sinr_capacity(
    dist_to_serving_m: float,
    dist_to_interferers_m: List[float],
    scenario: DeploymentScenario = DeploymentScenario.UMA,
    p_tx_dbm: float = 46.0,
    g_tx_dbi: float = 15.0,
    g_rx_ue_dbi: float = 0.0,
    serving_beamforming_gain_db: float = 18.0,
    interferer_beamforming_suppression_db: float = 10.0,
    carrier_freq_hz: float = 3.5e9,
    bandwidth_hz: float = 100e6,
    shadowing_std_dev_db: float = 7.8,
    body_loss_db: float = 3.0,
    noise_figure_db: float = 7.0,
    implementation_loss_factor: float = 0.65,
    ue_height_m: float = 1.5,
    bs_height_m: Optional[float] = None
) -> Tuple[float, float, float]:
    """
    Realistic 5G NR downlink SINR and throughput model using 3GPP pathloss.

    Returns:
        sinr_db, throughput_mbps, spectral_efficiency_bps_hz
    """

    # ── Serving signal power ──────────────────────────────────────────────────
    pl_serving_db = (
        pathloss_3gpp_nlos(scenario, dist_to_serving_m, carrier_freq_hz, ue_height_m, bs_height_m)
        + body_loss_db
        + np.random.normal(0.0, shadowing_std_dev_db)
    )

    s_dbm = (p_tx_dbm + g_tx_dbi + g_rx_ue_dbi + serving_beamforming_gain_db - pl_serving_db)
    s_mw = 10 ** (s_dbm / 10.0)

    # ── Aggregate interference ────────────────────────────────────────────────
    i_mw = 0.0
    for d_j in dist_to_interferers_m:
        pl_j_db = (
            pathloss_3gpp_nlos(scenario, d_j, carrier_freq_hz, ue_height_m, bs_height_m)
            + body_loss_db
            + np.random.normal(0.0, shadowing_std_dev_db)
        )

        p_rx_j_dbm = (p_tx_dbm + g_tx_dbi + g_rx_ue_dbi - interferer_beamforming_suppression_db - pl_j_db)
        i_mw += 10 ** (p_rx_j_dbm / 10.0)

    # ── Thermal noise ─────────────────────────────────────────────────────────
    # Telecom-standard thermal noise: N[dBm] = -174 + 10log10(BW) + NF
    n_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    n_mw = 10 ** (n_dbm / 10.0)

    # ── SINR & Capacity ───────────────────────────────────────────────────────
    sinr_linear = s_mw / (i_mw + n_mw)
    sinr_db = 10.0 * math.log10(sinr_linear)

    spectral_efficiency = implementation_loss_factor * math.log2(1.0 + sinr_linear)
    throughput_mbps = (bandwidth_hz * spectral_efficiency) / 1e6

    return sinr_db, throughput_mbps, spectral_efficiency


# ──────────────────────────────────────────────────────────────────────────────
# 4. NTN SINR & Capacity (LEO / MEO / GEO)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_ntn_sinr_capacity(
    slant_range_km: float,
    off_axis_angles_deg: List[float],
    eirp_dbw: float = 40.0,
    g_t_db: float = -15.5,
    freq_ghz: float = 2.0,
    bandwidth_hz: float = 40e6,
    weather_loss_db: float = 1.0,
    theta_3db_deg: float = 2.5,
    sll_db: float = 25.0,
    implementation_loss_factor: float = 0.65
) -> Tuple[float, float, float]:
    """
    Compute NTN SINR [dB] and Shannon beam capacity [Mbps].
    Uses 3GPP TR 38.821 Carrier-to-Noise Density (C/N0) accumulation.

    Returns:
        sinr_db, throughput_mbps, spectral_efficiency_bps_hz
    """

    fspl_db = _fspl_db_km_ghz(slant_range_km, freq_ghz)
    noise_bw_db = 10.0 * math.log10(bandwidth_hz)

    # ── Wanted Carrier Power to Noise Ratio ───────────────────────────────────
    cn0_dbhz = (eirp_dbw + g_t_db - fspl_db - weather_loss_db - K_DB)
    cn_db = cn0_dbhz - noise_bw_db
    s_linear = 10 ** (cn_db / 10.0)

    # ── Adjacent-beam interference accumulation ───────────────────────────────
    i_linear = 0.0
    for theta_off in off_axis_angles_deg:
        # Antenna pattern roll-off capped at Side-Lobe Level (SLL)
        roll_off_db = min(12.0 * (theta_off / theta_3db_deg) ** 2, sll_db)

        interferer_cn0_dbhz = (eirp_dbw - roll_off_db + g_t_db - fspl_db - weather_loss_db - K_DB)
        interferer_cn_db = interferer_cn0_dbhz - noise_bw_db
        i_linear += 10 ** (interferer_cn_db / 10.0)

    # ── SINR & Capacity ───────────────────────────────────────────────────────
    # Since signals are already relative to noise (C/N), denominator is (1 + I/N)
    sinr_linear = s_linear / (1.0 + i_linear)
    sinr_db = 10.0 * math.log10(sinr_linear)

    spectral_efficiency = implementation_loss_factor * math.log2(1.0 + sinr_linear)
    throughput_mbps = (bandwidth_hz * spectral_efficiency) / 1e6

    return sinr_db, throughput_mbps, spectral_efficiency


# ──────────────────────────────────────────────────────────────────────────────
# 5. TN Maximum Cell Radius (Bisection Search)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_max_tn_radius_km(
    p_tx_dbm: float,
    g_tx_dbi: float,
    g_rx_ue_dbi: float,
    carrier_freq_hz: float,
    bandwidth_hz: float,
    sinr_min_db: float,
    body_loss_db: float,
    scenario: DeploymentScenario = DeploymentScenario.UMA,
    interference_margin_db: float = 2.0,
    noise_figure_db: float = 7.0,
    ue_height_m: float = 1.5,
    bs_height_m: Optional[float] = None,
    tolerance_m: float = 1.0,
) -> float:
    """
    Derive the maximum TN cell radius [km] from a noise-limited link budget
    using bisection search against the correct 3GPP pathloss model.
    """
    if bs_height_m is None:
        bs_height_m = DEFAULT_H_BS[scenario]

    # Thermal noise floor [dBm]
    n_dbm = -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db

    # Minimum received power at UE [dBm] required to meet SINR
    p_rx_min_dbm = sinr_min_db + n_dbm + interference_margin_db

    # Maximum allowable path loss [dB]
    max_path_loss_db = (p_tx_dbm + g_tx_dbi + g_rx_ue_dbi - body_loss_db - p_rx_min_dbm)

    # ── Bisection over 3GPP Model ─────────────────────────────────────────────
    def _pl(d: float) -> float:
        return pathloss_3gpp_nlos(scenario, d, carrier_freq_hz, ue_height_m, bs_height_m)

    # Bounds check
    min_dist = 10.0
    max_dist = 5000.0 if scenario in [DeploymentScenario.UMA, DeploymentScenario.RMA] else 2000.0

    if _pl(min_dist) > max_path_loss_db:
        return 0.0
    if _pl(max_dist) <= max_path_loss_db:
        return max_dist / 1000.0

    # Search for crossing point
    d_lo, d_hi = min_dist, max_dist
    while (d_hi - d_lo) > tolerance_m:
        d_mid = (d_lo + d_hi) * 0.5
        if _pl(d_mid) < max_path_loss_db:
            d_lo = d_mid
        else:
            d_hi = d_mid

    return ((d_lo + d_hi) * 0.5) / 1000.0