"""
Physical, orbital, and system-level constants.

All values are in SI units unless explicitly noted.
"""

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_M_S: float = 2.998_292_458e8      # m/s  (exact IAU value)
BOLTZMANN_CONSTANT: float = 1.380_649e-23         # J/K  (exact SI value)
EARTH_RADIUS_M: float = 6_371_000.0               # m    mean spherical radius
EARTH_MU: float = 3.986_004_418e14                # m³/s²  standard gravitational parameter
EARTH_J2: float = 1.082_626_68e-3                 # dimensionless  second zonal harmonic
EARTH_ROTATION_RAD_S: float = 7.292_115e-5        # rad/s  sidereal rotation rate

# ---------------------------------------------------------------------------
# Frequency bands (Hz) – centre frequencies for reference
# ---------------------------------------------------------------------------
FREQ_KU_HZ: float = 12.0e9    # Ku-band  (downlink reference, Starlink Gen-1)
FREQ_KA_HZ: float = 26.5e9    # Ka-band  (Starlink Gen-2 / V2)
FREQ_S_HZ: float  =  2.4e9    # S-band   (mobile NTN, NB-IoT)
FREQ_L_HZ: float  =  1.6e9    # L-band   (Iridium / Inmarsat reference)

# ---------------------------------------------------------------------------
# Starlink (Phase-1 / Shell-1) reference constellation parameters
# Source: FCC filing SpaceX-SATMOD2016-00116
# ---------------------------------------------------------------------------
STARLINK_SHELL_1 = dict(
    name="Starlink-Shell-1",
    altitude_km=550.0,
    inclination_deg=53.0,
    num_planes=72,
    sats_per_plane=22,
    total_satellites=1584,
    phasing=1,
    freq_hz=FREQ_KU_HZ,
)

# Convenience alias for the default Starlink profile used in this project
STARLINK_DEFAULT = STARLINK_SHELL_1

# Additional reference constellations (kept for extensibility)
ONEWEB_SHELL_1 = dict(
    name="OneWeb-Shell-1",
    altitude_km=1200.0,
    inclination_deg=87.9,
    num_planes=18,
    sats_per_plane=40,
    total_satellites=720,
    phasing=1,
    freq_hz=FREQ_KU_HZ,
)

O3B_MEO = dict(
    name="O3b-MEO",
    altitude_km=8_062.0,
    inclination_deg=0.0,
    num_planes=1,
    sats_per_plane=20,
    total_satellites=20,
    phasing=1,
    freq_hz=FREQ_KA_HZ,
)

INMARSAT_GEO = dict(
    name="Inmarsat-GEO",
    altitude_km=35_786.0,
    inclination_deg=0.0,
    num_planes=1,
    sats_per_plane=1,
    total_satellites=1,
    phasing=0,
    freq_hz=FREQ_L_HZ,
)

# ---------------------------------------------------------------------------
# Orbit altitude bands (km) – used for classification
# ---------------------------------------------------------------------------
LEO_ALT_RANGE_KM = (200.0, 2_000.0)
MEO_ALT_RANGE_KM = (2_000.0, 35_786.0)
GEO_ALT_KM       = 35_786.0          # geostationary altitude
GEO_TOLERANCE_KM = 200.0             # ± km considered GEO

# ---------------------------------------------------------------------------
# Link-budget defaults (can be overridden via config)
# ---------------------------------------------------------------------------
NOISE_TEMPERATURE_K: float = 290.0        # K  (standard reference temperature)
ANTENNA_EFFICIENCY: float  = 0.55         # dimensionless
SYSTEM_NOISE_FIGURE_DB: float = 3.0       # dB  receiver noise figure
ATMOSPHERIC_LOSS_DB: float = 0.5          # dB  clear-sky margin (Ku-band, 10° el.)
RAIN_FADE_MARGIN_DB: float = 3.0          # dB  ITU-R P.618 temperate climate

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_TIME_STEP_S: float = 60.0         # s   propagation / snapshot cadence
DEFAULT_MIN_ELEVATION_DEG: float = 25.0   # deg minimum elevation angle for service
DEFAULT_EPOCH: str = "2024-01-01T00:00:00"  # ISO-8601 UTC