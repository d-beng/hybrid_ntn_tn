import math
import numpy as np
from typing import List, Tuple

# ─────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────
C_M_S   = 299_792_458.0   # speed of light [m/s]
K_B     = 1.380649e-23    # Boltzmann constant [J/K]
T_SYS_K = 290.0           # reference system noise temperature [K]
K_DB = -228.6  # Boltzmann constant [dBW/K/Hz]


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _fspl_db(distance_m: float, freq_hz: float) -> float:
    """Free-Space Path Loss [dB].  distance in metres, freq in Hz."""
    d = max(distance_m, 1.0)
    return (20 * math.log10(d)
            + 20 * math.log10(freq_hz)
            + 20 * math.log10(4 * math.pi / C_M_S))


def _fspl_db_km_ghz(distance_km: float, freq_ghz: float) -> float:
    """FSPL [dB] using the telecom shorthand constant 92.45.
    FSPL = 20·log10(d_km) + 20·log10(f_GHz) + 92.45
    Derivation: 20·log10(1e3) + 20·log10(1e9) + 20·log10(4π/c) ≈ 92.45
    """
    return (20 * math.log10(max(distance_km, 1e-3))
            + 20 * math.log10(max(freq_ghz, 1e-6))
            + 92.45)


# ─────────────────────────────────────────────
# 1. TN SINR & Capacity  (5G NR)
# ─────────────────────────────────────────────

def calculate_tn_sinr_capacity(
    dist_to_serving_m: float,
    dist_to_interferers_m: List[float],
    p_tx_dbm: float = 43.0,          # BS transmit power [dBm]
    g_tx_dbi: float = 15.0,          # BS antenna gain [dBi]
    g_rx_ue_dbi: float = 0.0,        # UE receive gain [dBi]
    carrier_freq_hz: float = 3.5e9,  # carrier frequency [Hz]
    bandwidth_hz: float = 100e6,     # channel bandwidth [Hz]
    shadowing_std_dev_db: float = 8.0,
    body_loss_db: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute 5G TN SINR [dB] and Shannon capacity [Mbps].

    Interference model
    ------------------
    All cells passed in dist_to_interferers_m are assumed to operate on the
    SAME frequency channel (full reuse = 1).  For a 3-cell reuse pattern,
    pre-filter the list in the caller before passing it here.

    Shadowing
    ---------
    Each path draws an independent log-normal shadow sample.  Spatial
    correlation is not modelled (known simplification — raises SINR variance
    slightly vs. reality).
    """
    # ── Serving cell received power ──────────────────────────────────────
    pl_serving_db = (_fspl_db(dist_to_serving_m, carrier_freq_hz)
                     + body_loss_db
                     + np.random.normal(0.0, shadowing_std_dev_db))

    s_dbm  = p_tx_dbm + g_tx_dbi + g_rx_ue_dbi - pl_serving_db
    s_mw   = 10 ** (s_dbm / 10.0)

    # ── Aggregate interference [mW linear] ───────────────────────────────
    i_mw = 0.0
    for d_j in dist_to_interferers_m:
        pl_j_db = (_fspl_db(d_j, carrier_freq_hz)
                   + body_loss_db
                   + np.random.normal(0.0, shadowing_std_dev_db))
        p_rx_j_dbm = p_tx_dbm + g_tx_dbi + g_rx_ue_dbi - pl_j_db
        i_mw += 10 ** (p_rx_j_dbm / 10.0)

    # ── Thermal noise [mW] ───────────────────────────────────────────────
    # N = k_B · T · BW  [W]  →  ×1000 → [mW]
    n_mw = K_B * T_SYS_K * bandwidth_hz * 1_000.0

    # ── SINR & capacity ──────────────────────────────────────────────────
    sinr_linear  = s_mw / (i_mw + n_mw)
    sinr_db      = 10.0 * math.log10(sinr_linear)
    capacity_mbps = bandwidth_hz * math.log2(1.0 + sinr_linear) / 1e6

    return sinr_db, capacity_mbps


# ─────────────────────────────────────────────
# 2. NTN SINR & Capacity  (LEO / MEO / GEO)
# ─────────────────────────────────────────────
def calculate_ntn_sinr_capacity_old(
    slant_range_km: float,
    off_axis_angles_deg: List[float],
    eirp_dbw: float = 40.0,          # satellite EIRP [dBW]  (= P_tx + G_tx on board)
    g_t_db: float = 10.0,            # user terminal G/T [dB/K]  ← ratio, NOT G+T
    freq_ghz: float = 12.0,          # downlink carrier [GHz]
    bandwidth_hz: float = 250e6,     # beam bandwidth [Hz]
    weather_loss_db: float = 1.0,    # atm + rain loss [dB]
    theta_3db_deg: float = 2.5,      # beam 3 dB half-angle [deg]
    sll_db: float = 25.0,            # side-lobe level isolation [dB]
) -> Tuple[float, float]:
    """
    Compute NTN SINR [dB] and Shannon beam capacity [Mbps].

    Link budget (ITU / 3GPP TR 38.821)
    ------------------------------------
    C/N₀ [dB·Hz] = EIRP [dBW] + G/T [dB/K] − FSPL [dB] − losses [dB] − k [dBW/K·Hz]
    C/N  [dB]    = C/N₀ − 10·log10(BW)
    SINR [dB]    = C / (I + N)   computed in linear watts

    G/T definition
    --------------
    G/T [dB/K] = G_rx [dBi] − 10·log10(T_sys [K])
    It is a RATIO.  The noise temperature is already embedded in G/T and in
    the thermal noise floor N = k_B · T_sys · BW.  Do NOT add T_sys again
    to G/T — that was the original bug.

    Interference model (3GPP TR 38.821 §6.1)
    -----------------------------------------
    Adjacent beams of the SAME satellite on the same frequency sub-band are
    attenuated by the beam antenna pattern:
        G(θ) = G_max − min(12·(θ/θ_3dB)², SLL)   [dB]
    where θ is the off-axis angle at the satellite between the wanted beam
    centre and the interfering beam centre.
    Approximation: same slant range is used for all intra-satellite beams
    (valid because the ground separation between beams << slant range for LEO).
    """
    # ── Free-space path loss [dB] ─────────────────────────────────────────
    fspl_db = _fspl_db_km_ghz(slant_range_km, freq_ghz)

    # ── Wanted signal power [W] ───────────────────────────────────────────
    # C [dBW] = EIRP [dBW] + G/T [dB/K] − FSPL [dB] − L_weather [dB]
    #           − 10·log10(k_B [W/K·Hz]) − 10·log10(BW [Hz])
    # But we compute linearly to sum I and N properly:
    #
    # Received C/N₀ in linear:
    #   s_w = 10^( (EIRP + G/T − FSPL − L) / 10 ) · k_B   [W/Hz × Hz = W]
    # Simpler: treat G/T as net receive gain after noise figure, then noise
    # separately.
    #
    # Numerically correct form:
    #   s_w = 10^( (EIRP_dbw + g_t_db − fspl_db − weather_loss_db) / 10 )
    # This gives C·(G/T) in [W·K⁻¹]; multiplying by k_B gives C/N₀ in [W].
    # We keep it as [W] and set noise as k_B·T_sys·BW so SINR = s_w / (i_w + n_w).
    #
    # Note: g_t_db [dB/K] encodes T_sys implicitly.  We use T_SYS_K=290 K as
    # the noise reference for n_w, consistent with the standard 290 K default.

    s_dbw = eirp_dbw + g_t_db - fspl_db - weather_loss_db
    s_w   = 10 ** (s_dbw / 10.0)

    # ── Adjacent-beam interference [W] ────────────────────────────────────
    # Same slant range assumed for all intra-satellite beams (documented approx).
    i_w = 0.0
    for theta_off in off_axis_angles_deg:
        # 3GPP parabolic beam pattern with SLL floor
        roll_off_db   = min(12.0 * (theta_off / theta_3db_deg) ** 2, sll_db)
        p_adj_dbw     = (eirp_dbw - roll_off_db) + g_t_db - fspl_db - weather_loss_db
        i_w          += 10 ** (p_adj_dbw / 10.0)

    # ── Thermal noise [W] ─────────────────────────────────────────────────
    # N = k_B · T_sys · BW
    n_w = K_B * T_SYS_K * bandwidth_hz

    # ── SINR & capacity ───────────────────────────────────────────────────
    sinr_linear   = s_w / (i_w + n_w)
    sinr_db       = 10.0 * math.log10(sinr_linear)
    capacity_mbps = bandwidth_hz * math.log2(1.0 + sinr_linear) / 1e6

    return sinr_db, capacity_mbps






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
):

    fspl_db = _fspl_db_km_ghz(slant_range_km, freq_ghz)

    # -----------------------------------------
    # Carrier-to-noise density ratio
    # -----------------------------------------
    cn0_dbhz = (
        eirp_dbw
        + g_t_db
        - fspl_db
        - weather_loss_db
        - K_DB
    )

    # -----------------------------------------
    # Convert to carrier/noise over bandwidth
    # -----------------------------------------
    noise_bw_db = 10 * math.log10(bandwidth_hz)

    cn_db = cn0_dbhz - noise_bw_db

    # Desired carrier power relative to noise
    s_linear = 10 ** (cn_db / 10.0)

    # -----------------------------------------
    # Interference accumulation
    # -----------------------------------------
    i_linear = 0.0

    for theta_off in off_axis_angles_deg:

        roll_off_db = min(
            12.0 * (theta_off / theta_3db_deg) ** 2,
            sll_db,
        )

        interferer_cn0_dbhz = (
            eirp_dbw
            - roll_off_db
            + g_t_db
            - fspl_db
            - weather_loss_db
            - K_DB
        )

        interferer_cn_db = (
            interferer_cn0_dbhz
            - noise_bw_db
        )

        i_linear += 10 ** (interferer_cn_db / 10.0)

    sinr_linear = s_linear / (1.0 + i_linear)

    sinr_db = 10 * math.log10(sinr_linear)

    capacity_mbps = (
        bandwidth_hz
        * math.log2(1 + sinr_linear)
        / 1e6
    )

    return sinr_db, capacity_mbps

# ─────────────────────────────────────────────
# 3. TN maximum cell radius  (link budget)
# ─────────────────────────────────────────────

def calculate_max_tn_radius_km(
    p_tx_dbm: float,
    g_tx_dbi: float,
    g_rx_ue_dbi: float,
    carrier_freq_hz: float,
    bandwidth_hz: float,
    sinr_min_db: float,
    body_loss_db: float,
    interference_margin_db: float = 2.0,   # explicit named margin (was hardcoded +2)
) -> float:
    """
    Derive the maximum TN cell radius [km] from a noise-limited link budget.

    The interference_margin_db accounts for the residual ICI not captured by
    the explicit interferer list.  Default 2 dB is a standard planning value
    (3GPP TR 38.913).  Expose it so the caller can tune it from config.

    Steps
    -----
    1. Compute thermal noise floor N [dBm].
    2. Minimum received power P_rx_min = SINR_min + N + I_margin.
    3. Maximum allowed path loss PL_max = EIRP_UL − P_rx_min − body loss.
    4. Invert FSPL formula to get d_max.
    """
    # Thermal noise floor [dBm]
    n_w   = K_B * T_SYS_K * bandwidth_hz          # [W]
    n_dbm = 10.0 * math.log10(n_w * 1_000.0)      # [dBm]

    # Minimum received power at UE [dBm]
    p_rx_min_dbm = sinr_min_db + n_dbm + interference_margin_db

    # Maximum allowable path loss [dB]
    pl_max_db = (p_tx_dbm + g_tx_dbi + g_rx_ue_dbi
                 - body_loss_db
                 - p_rx_min_dbm)

    # Invert FSPL: PL = 20·log10(d) + 20·log10(f) + 20·log10(4π/c)
    const_term = (20 * math.log10(carrier_freq_hz)
                  + 20 * math.log10(4 * math.pi / C_M_S))
    log10_d    = (pl_max_db - const_term) / 20.0
    d_max_m    = 10 ** log10_d

    return d_max_m / 1_000.0   # [km]