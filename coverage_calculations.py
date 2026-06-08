from hybrid_ntn_optimizer.link_budget.sinr import DeploymentScenario, calculate_max_tn_radius_km

def run_coverage_test():
    print("======================================================================")
    print(" 3GPP Maximum Cell Radius: Downlink (DL) vs. Uplink (UL) Bottleneck")
    print("======================================================================\n")

    # ── 1. Shared Constants ───────────────────────────────────────────────────
    sinr_min_db = -5.0             # Minimum acceptable SINR for connection
    body_loss_db = 3.0             # Human body blockage
    interference_margin_db = 2.0   # Background interference buffer
    ue_height_m = 1.5              # Standard human holding a phone

    # ── 2. Hardware Profiles ──────────────────────────────────────────────────
    # User Equipment (Smartphone) Hardware Limits
    ue_p_tx_dbm = 23.0             # UE Max Transmit Power (0.2 Watts)
    ue_g_ant_dbi = 0.0             # UE Antenna Gain (Omnidirectional)
    ue_nf_db = 9.0                 # UE Receiver Noise Figure (usually worse than BS)

    # ── 3. Scenario-Specific Parameters ───────────────────────────────────────
    # We dynamically set Frequency, BW, and BS Hardware based on 3GPP standards
    test_cases = [
        {
            "scenario": DeploymentScenario.UMA,
            "freq_hz": 3.5e9,        # 3.5 GHz (Mid-band C-Band)
            "bandwidth_hz": 100e6,   # 100 MHz 
            "bs_p_tx_dbm": 46.0,     # 40 Watts (Macro)
            "bs_g_ant_dbi": 17.0,    # High gain sector antenna
            "bs_nf_db": 5.0          # High quality BS receiver
        },
        {
            "scenario": DeploymentScenario.UMI,
            "freq_hz": 3.5e9,        # 3.5 GHz
            "bandwidth_hz": 100e6,   # 100 MHz
            "bs_p_tx_dbm": 38.0,     # 6.3 Watts (Street lamp Small Cell)
            "bs_g_ant_dbi": 10.0,    # Medium gain antenna
            "bs_nf_db": 5.0
        },
        {
            "scenario": DeploymentScenario.RMA,
            "freq_hz": 0.7e9,        # 700 MHz (Low-band for extreme distance)
            "bandwidth_hz": 20e6,    # 20 MHz (Restricted bandwidth for low frequencies)
            "bs_p_tx_dbm": 46.0,     # 40 Watts
            "bs_g_ant_dbi": 15.0,    # Macro antenna
            "bs_nf_db": 5.0
        },
        {
            "scenario": DeploymentScenario.INH,
            "freq_hz": 4e9,          # 4 GHz 
            "bandwidth_hz": 100e6,   # 100 MHz (Indoor Hotspot)
            "bs_p_tx_dbm": 24.0,     # 250 milliWatts (Ceiling router)
            "bs_g_ant_dbi": 5.0,     # Low gain omni antenna
            "bs_nf_db": 7.0          # Cheaper indoor receiver
        },
        {
            "scenario": DeploymentScenario.INF_SH,
            "freq_hz": 28e9,         # 28 GHz (mmWave for Factory Automation)
            "bandwidth_hz": 400e6,   # 400 MHz (Massive capacity)
            "bs_p_tx_dbm": 30.0,     # 1 Watt
            "bs_g_ant_dbi": 15.0,    # Directional mmWave array
            "bs_nf_db": 7.0
        }
    ]

    # ── 4. Run the Bisection Calculator ───────────────────────────────────────
    for tc in test_cases:
        scenario = tc["scenario"]
        
        # --- DOWNLINK (BS Transmitting -> UE Receiving) ---
        radius_dl_km = calculate_max_tn_radius_km(
            p_tx_dbm=tc["bs_p_tx_dbm"],
            g_tx_dbi=tc["bs_g_ant_dbi"],
            g_rx_ue_dbi=ue_g_ant_dbi,
            carrier_freq_hz=tc["freq_hz"],
            bandwidth_hz=tc["bandwidth_hz"],
            sinr_min_db=sinr_min_db,
            body_loss_db=body_loss_db,
            scenario=scenario,
            interference_margin_db=interference_margin_db,
            noise_figure_db=ue_nf_db, # UE is the receiver
            ue_height_m=ue_height_m
        )

        # --- UPLINK (UE Transmitting -> BS Receiving) ---
        radius_ul_km = calculate_max_tn_radius_km(
            p_tx_dbm=ue_p_tx_dbm,          # The Phone is transmitting
            g_tx_dbi=ue_g_ant_dbi,         # Phone antenna gain
            g_rx_ue_dbi=tc["bs_g_ant_dbi"],# Base Station is receiving
            carrier_freq_hz=tc["freq_hz"],
            bandwidth_hz=tc["bandwidth_hz"],
            sinr_min_db=sinr_min_db,
            body_loss_db=body_loss_db,
            scenario=scenario,
            interference_margin_db=interference_margin_db,
            noise_figure_db=tc["bs_nf_db"], # BS is the receiver
            ue_height_m=ue_height_m
        )

        # ── 5. Print Results ──────────────────────────────────────────────────
        # The true effective radius is the bottleneck (the smaller of the two)
        effective_radius = min(radius_dl_km, radius_ul_km)

        print(f"Scenario: {scenario.value}")
        print(f"  Carrier    : {tc['freq_hz'] / 1e9} GHz  |  Bandwidth: {tc['bandwidth_hz'] / 1e6} MHz")
        
        if radius_dl_km > 0:
            print(f"  DL Radius  : {radius_dl_km:.3f} km  (BS Tx: {tc['bs_p_tx_dbm']} dBm)")
        else:
            print(f"  DL Radius  : FAILED (Budget too tight) radius_dl_km={radius_dl_km:.3f} km")

        if radius_ul_km > 0:
            print(f"  UL Radius  : {radius_ul_km:.3f} km  (UE Tx: {ue_p_tx_dbm} dBm)")
        else:
            print(f"  UL Radius  : FAILED (Budget too tight)")

        print(f"  ► EFFECTIVE BOTTLENECK RADIUS: {effective_radius:.3f} km\n")

if __name__ == "__main__":
    run_coverage_test()