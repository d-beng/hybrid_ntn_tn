import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import random
import h3
from pathlib import Path
import streamlit.components.v1 as components
from omegaconf import OmegaConf, DictConfig
from shapely.geometry import shape, mapping
import plotly.express as px

# Import the team's modular architecture
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.core.types import WalkerParameters, OrbitType
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.traffic.profiles import generate_users
from hybrid_ntn_optimizer.terrestrial.coverage import generate_terrestrial_network
from hybrid_ntn_optimizer.simulation.full_pipeline import run_daily_mobility_simulation
from hybrid_ntn_optimizer.visualization.plots import build_h3_geojson

# ==========================================
# CONFIGURATION FILE LOADING
# ==========================================
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

def _load_cfg(path: Path) -> DictConfig:
    if not path.exists(): return OmegaConf.create({})
    return OmegaConf.load(path)

def _to_float(value, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float(default)

def _to_int(value, default: int = 0) -> int:
    try: return int(round(float(value)))
    except (TypeError, ValueError): return int(default)

def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

base_cfg_defaults = _load_cfg(CONFIG_DIR / "base.yaml")
constellation_cfg_defaults = _load_cfg(CONFIG_DIR / "constellation.yaml")
scenario_yaml_cfg = _load_cfg(CONFIG_DIR / "scenario" / "ontario_full.yaml")
population_yaml_cfg = _load_cfg(CONFIG_DIR / "population" / "ontario_demographics.yaml")
terrestrial_yaml_cfg = _load_cfg(CONFIG_DIR / "terrestrial" / "5g_base.yaml")
cost_yaml_cfg = _load_cfg(CONFIG_DIR / "cost.yaml")
mobility_yaml_cfg = _load_cfg(CONFIG_DIR / "mobility.yaml")
optimization_yaml_cfg = _load_cfg(CONFIG_DIR / "optimization.yaml")

ontario_yaml = OmegaConf.to_container(scenario_yaml_cfg, resolve=True)
if "geojson_geometry" in ontario_yaml:
    ONTARIO_GEOM = shape(ontario_yaml["geojson_geometry"])
    if not ONTARIO_GEOM.is_valid:
        ONTARIO_GEOM = ONTARIO_GEOM.buffer(0)
        ontario_yaml["geojson_geometry"] = mapping(ONTARIO_GEOM)
    LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = ONTARIO_GEOM.bounds
    center_lat, center_lon = (LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2
else:
    center_lat, center_lon = 45.0, -80.0
    ONTARIO_GEOM = None

DEFAULT_TOTAL_USERS = _to_int(population_yaml_cfg.get("total_city_users", 700)) + _to_int(population_yaml_cfg.get("total_rural_users", 300))
DEFAULT_CITY_RATIO = round(_clamp(_to_int(population_yaml_cfg.get("total_city_users", 700)) / DEFAULT_TOTAL_USERS if DEFAULT_TOTAL_USERS > 0 else 0.7, 0.1, 0.9), 1)
DEFAULT_TN_SHADOWING = float(_clamp(
    _to_float(OmegaConf.select(terrestrial_yaml_cfg, "shadowing_std_dev_db", default=8.0), 8.0),
    0.0,
    20.0
))

DEFAULT_TN_BODY_LOSS = float(_clamp(
    _to_float(OmegaConf.select(terrestrial_yaml_cfg, "body_loss_db", default=3.0), 3.0),
    0.0,
    15.0
))

DEFAULT_EVENING_PEAK_HOUR = float(_clamp(
    _to_float(
        OmegaConf.select(
            population_yaml_cfg,
            "traffic.diurnal_curve.evening_peak.center_hour",
            default=20.0
        ),
        20.0
    ),
    16.0,
    23.0
))

DEFAULT_SAT_ALTITUDE = float(_clamp(
    _to_float(OmegaConf.select(constellation_cfg_defaults, "constellation.altitude_km", default=550.0), 550.0),
    300.0,
    1500.0,
))
DEFAULT_TOTAL_SATS = _to_int(OmegaConf.select(constellation_cfg_defaults, "constellation.total_satellites", default=1584), 1584)
DEFAULT_NUM_PLANES = _to_int(OmegaConf.select(constellation_cfg_defaults, "constellation.num_planes", default=72), 72)
DEFAULT_NTN_BW_MHZ = _clamp(
    _to_int(_to_float(OmegaConf.select(constellation_cfg_defaults, "constellation.bandwidth_hz", default=300e6), 300e6) / 1e6, 300),
    10,
    1000,
)
DEFAULT_SAT_EIRP = float(_clamp(
    _to_float(OmegaConf.select(constellation_cfg_defaults, "constellation.eirp_dbw", default=50.0), 50.0),
    20.0,
    70.0,
))

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="TN/LEO Network Digital Twin", layout="wide")
st.title("TN/LEO Network Digital Twin")
st.markdown(
    "K-Means places fixed 5G/TN infrastructure in dense areas. Users reassociate every time step using physical TN coverage, SINR, bandwidth, and backhaul. "
    "Spillover is OFF: users inside TN coverage are served by TN or dropped; users outside TN coverage use LEO as primary access."
)
st.info(
    "K-Means clusters/base stations are created once from the generated home-user distribution. "
    "They are not recreated every hour; STEPS mobility changes user positions and runtime network states."
)

# ==========================================
# SIDEBAR: USER INPUTS & TOGGLES
# ==========================================
st.sidebar.header("Presentation Mode")
VISUAL_MODE = st.sidebar.radio(
    "Select Visualization Style:",
    ["Geometric (Voronoi)", "Heatmap (Planhub)"],
    help="Changes map visuals INSTANTLY without recalculating physics."
)
st.sidebar.markdown("---")

st.sidebar.header("Simulation Parameters")
TOTAL_USERS = st.sidebar.slider("Total Simulated Users", min_value=100, max_value=5000, value=max(100, DEFAULT_TOTAL_USERS), step=100)
CITY_RATIO = st.sidebar.slider("City vs Rural Ratio", min_value=0.1, max_value=0.9, value=DEFAULT_CITY_RATIO, step=0.1)

st.sidebar.subheader("Infrastructure")
TN_CITY_THRESHOLD = st.sidebar.slider("Urban Discovery Threshold", 2, 100, 50)
TN_USERS_PER_TOWER = st.sidebar.slider("Target Users per Tower", 5, 50, 20)
TN_BS_CAPACITY_GBPS = st.sidebar.slider("Tower Backhaul (Gbps)", 1, 100, 50)
TN_BS_CAPACITY_MBPS = TN_BS_CAPACITY_GBPS * 1000
TN_BW_MHZ = st.sidebar.slider("BS Bandwidth (MHz)", 10, 800, 400, 10)

with st.sidebar.expander("Advanced 5G RF Parameters"):
    TN_RADIUS_MIN_KM = st.sidebar.slider("Minimum BS Coverage Radius (km)", 1.0, 3.0, 1.0, 0.1)
    TN_RADIUS_MAX_KM = st.sidebar.slider("Maximum BS Coverage Radius (km)", 1.0, 5.0, 5.0, 0.1)
    if TN_RADIUS_MAX_KM < TN_RADIUS_MIN_KM:
        TN_RADIUS_MAX_KM = TN_RADIUS_MIN_KM
        st.warning("Maximum BS radius was raised to match the minimum radius.")
    TN_MAX_EXTENT = TN_RADIUS_MAX_KM
    TN_P_TX = st.slider("BS Transmit Power (dBm)", 20.0, 60.0, 43.0, 1.0)
    TN_G_TX = st.slider("BS Antenna Gain (dBi)", 0.0, 30.0, 15.0, 1.0)
    TN_G_RX = st.slider("UE Receive Gain (dBi)", -10.0, 10.0, 0.0, 1.0)
    TN_FREQ_GHZ = st.slider("Carrier Frequency (GHz)", 0.5, 6.0, 3.5, 0.1)
    TN_SINR_MIN = st.slider("Min SINR (dB)", -10.0, 10.0, -5.0, 0.5)
    TN_SHADOWING = st.slider(
        "Shadowing Std Dev (dB)",
        min_value=0.0,
        max_value=20.0,
        value=DEFAULT_TN_SHADOWING,
        step=0.5,
    )

    TN_BODY_LOSS = st.slider(
        "Body/Penetration Loss (dB)",
        min_value=0.0,
        max_value=15.0,
        value=DEFAULT_TN_BODY_LOSS,
        step=0.5,
    )

st.sidebar.subheader("LEO Constellation")
with st.sidebar.expander("LEO Constellation Parameters", expanded=False):
    st.caption(
        "Defaults come from constellation.yaml. Leaving these controls unchanged preserves the YAML values; "
        "changing one applies a temporary GUI override and reruns the simulation."
    )
    SAT_ALTITUDE = st.slider(
        "Satellite Altitude (km)",
        min_value=300.0,
        max_value=1500.0,
        value=float(DEFAULT_SAT_ALTITUDE),
        step=50.0,
    )
    satellite_options = sorted(set([72, 324, 648, 1584, 4000, int(DEFAULT_TOTAL_SATS)]))
    TOTAL_SATS = st.select_slider(
        "Total Satellites in Constellation",
        options=satellite_options,
        value=int(DEFAULT_TOTAL_SATS),
    )
    NTN_BW_MHZ = st.slider(
        "NTN Beam Bandwidth (MHz)",
        min_value=10,
        max_value=1000,
        value=int(DEFAULT_NTN_BW_MHZ),
        step=10,
    )
    SAT_EIRP = st.slider(
        "Satellite EIRP (dBW)",
        min_value=20.0,
        max_value=70.0,
        value=float(DEFAULT_SAT_EIRP),
        step=1.0,
    )
    st.caption(f"Orbital planes remain YAML-backed: {DEFAULT_NUM_PLANES}")

st.sidebar.markdown("---")
st.sidebar.header("Map Layers")
SHOW_KMEANS_SHAPES = st.sidebar.checkbox("Show K-Means/Voronoi cluster shapes", value=True)
SHOW_TN_CIRCLES = st.sidebar.checkbox("Show TN coverage circles", value=True)
SHOW_LEO_HEXAGONS = st.sidebar.checkbox("Show active LEO hexagons", value=True)

st.sidebar.subheader("Traffic Profiles")
use_light = st.sidebar.checkbox("Light Users", value=True)
use_medium = st.sidebar.checkbox("Medium Users", value=True)
use_heavy = st.sidebar.checkbox("Heavy Users", value=True)
SIM_DURATION = st.sidebar.slider("Simulation Duration (s)", 3600, 86400, 86400, 3600)
TIME_STEP = st.sidebar.slider("Time Step (s)", 600, 3600, 3600, 600)
EVENING_PEAK_HOUR = st.sidebar.slider("Evening Peak Hour", 16.0, 23.0, DEFAULT_EVENING_PEAK_HOUR, 0.5)

# ==========================================
# MAP HELPERS
# ==========================================
def _boundary_to_feature(boundary, properties):
    if not boundary or len(boundary) < 3:
        return None
    coords = [[float(lon), float(lat)] for lat, lon in boundary]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": properties,
    }


def build_boundary_geojson(base_stations, attr_name: str):
    features = []
    for bs in base_stations:
        boundary = getattr(bs, attr_name, [])
        feature = _boundary_to_feature(boundary, {
            "tower_id": int(bs.bs_id),
            "radius_km": float(getattr(bs, "coverage_radius_km", 0.0) or 0.0),
            "assigned_users": int(getattr(bs, "assigned_user_count", 0) or 0),
        })
        if feature is not None:
            features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def build_voronoi_boundary_geojson(base_stations):
    return build_boundary_geojson(base_stations, "voronoi_boundary")


def build_coverage_boundary_geojson(base_stations):
    return build_boundary_geojson(base_stations, "coverage_boundary")

def get_boundary_coords(geom):
    if not geom: return [], []
    x_all, y_all = [], []
    polygons = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polygons:
        x, y = poly.exterior.xy
        x_all.extend(list(x) + [None])
        y_all.extend(list(y) + [None])
    return x_all, y_all

# ==========================================
# MINIMAL MAP HOVER HELPERS
# ==========================================
def _user_number_hover(rows):
    """Return only the simulated user number, e.g. User565."""
    return [f"User{int(u.get('User_ID', -1))}" for u in rows]


def _tn_radius_hover(base_stations):
    """Return only the per-station deployment radius shown by the orange circle."""
    return [
        f"Dynamic radius: {float(getattr(bs, 'coverage_radius_km', 0.0) or 0.0):.2f} km"
        for bs in base_stations
    ]


# ==========================================
# DYNAMIC MAP RENDERER
# ==========================================
def render_custom_dashboard_animation(region, users, base_stations, beam_data, user_data, duration_s, time_step_s, filename, visual_mode):
    hex_geojson = build_h3_geojson(region.cells)
    voronoi_geojson = build_voronoi_boundary_geojson(base_stations)
    coverage_geojson = build_coverage_boundary_geojson(base_stations)
    time_steps = list(range(0, duration_s + time_step_s, time_step_s))
    all_h3_ids = [cell.h3_id for cell in region.cells]
    
    fig = go.Figure()

    # TRACE 0: SATELLITE BEAMS / LEO H3 CELLS
    beam_opacity = 0.65 if SHOW_LEO_HEXAGONS else 0.0
    fig.add_trace(go.Choroplethmapbox(
        geojson=hex_geojson, locations=all_h3_ids, z=[0]*len(all_h3_ids),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0, 200, 90, 0.28)"]],
        zmin=0, zmax=1, marker_opacity=beam_opacity, showscale=False, hoverinfo="skip",
        name="Active LEO Hexagons"
    ))

    initial_users = [u for u in user_data if u["Hour"] == "Hour 0.0"]
    tn_u = [u for u in initial_users if u["State"] == "TN"]
    leo_u = [u for u in initial_users if u["State"] == "LEO"]
    drop_u = [u for u in initial_users if u["State"] == "DROPPED"]

    if visual_mode == "Heatmap (Planhub)":
        fig.add_trace(go.Densitymapbox(lat=[u["Lat"] for u in tn_u], lon=[u["Lon"] for u in tn_u], z=[1]*len(tn_u), radius=15, colorscale=[[0, "rgba(0,191,255,0)"], [1, "rgba(0,191,255,0.8)"]], showscale=False, showlegend=False))
        fig.add_trace(go.Densitymapbox(lat=[u["Lat"] for u in leo_u], lon=[u["Lon"] for u in leo_u], z=[1]*len(leo_u), radius=15, colorscale=[[0, "rgba(0,200,90,0)"], [1, "rgba(0,200,90,0.8)"]], showscale=False, showlegend=False))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=[u["Lat"] for u in tn_u], lon=[u["Lon"] for u in tn_u],
            customdata=_user_number_hover(tn_u),
            mode='markers', marker=dict(size=5, color='deepskyblue'),
            name="5G Users", showlegend=True,
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[u["Lat"] for u in leo_u], lon=[u["Lon"] for u in leo_u],
            customdata=_user_number_hover(leo_u),
            mode='markers', marker=dict(size=5, color='limegreen'),
            name="LEO / Satellite Users", showlegend=True,
            hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.add_trace(go.Scattermapbox(
        lat=[u["Lat"] for u in drop_u], lon=[u["Lon"] for u in drop_u],
        customdata=_user_number_hover(drop_u),
        mode='markers', marker=dict(size=4, color='red'),
        name="Outage", showlegend=True,
        hovertemplate="%{customdata}<extra></extra>",
    ))

    if visual_mode == "Heatmap (Planhub)":
        fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='markers', marker=dict(size=10, color='deepskyblue'), name="5G Coverage Area", showlegend=True))
        fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='markers', marker=dict(size=10, color='limegreen'), name="Sat Coverage Area", showlegend=True))
    else:
        fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='markers', showlegend=False))
        fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='markers', showlegend=False))

    fig.add_trace(go.Scattermapbox(
        lat=[bs.lat for bs in base_stations], lon=[bs.lon for bs in base_stations],
        customdata=_tn_radius_hover(base_stations),
        mode='markers', marker=dict(size=12, color='orange', opacity=0.95),
        name='TN Base Stations', showlegend=True,
        hovertemplate="%{customdata}<extra></extra>",
    ))

    frames = []
    slider_steps = []
    for t_s in time_steps:
        hour_str = f"Hour {t_s / 3600.0:.1f}"
        active_beams = [b["h3_id"] for b in beam_data if b["time_s"] == t_s]
        frame_z = [1 if h3_id in active_beams else 0 for h3_id in all_h3_ids]
        
        frame_users = [u for u in user_data if u["Hour"] == hour_str]
        tn_f = [u for u in frame_users if u["State"] == "TN"]
        leo_f = [u for u in frame_users if u["State"] == "LEO"]
        drop_f = [u for u in frame_users if u["State"] == "DROPPED"]

        frame_data = [go.Choroplethmapbox(z=frame_z)]
        if visual_mode == "Heatmap (Planhub)":
            frame_data.append(go.Densitymapbox(lat=[u["Lat"] for u in tn_f], lon=[u["Lon"] for u in tn_f]))
            frame_data.append(go.Densitymapbox(lat=[u["Lat"] for u in leo_f], lon=[u["Lon"] for u in leo_f]))
        else:
            frame_data.append(go.Scattermapbox(
                lat=[u["Lat"] for u in tn_f], lon=[u["Lon"] for u in tn_f],
                customdata=_user_number_hover(tn_f),
                hovertemplate="%{customdata}<extra></extra>",
            ))
            frame_data.append(go.Scattermapbox(
                lat=[u["Lat"] for u in leo_f], lon=[u["Lon"] for u in leo_f],
                customdata=_user_number_hover(leo_f),
                hovertemplate="%{customdata}<extra></extra>",
            ))
            
        frame_data.append(go.Scattermapbox(
            lat=[u["Lat"] for u in drop_f], lon=[u["Lon"] for u in drop_f],
            customdata=_user_number_hover(drop_f),
            hovertemplate="%{customdata}<extra></extra>",
        ))
        frame_data.append(go.Scattermapbox(lat=[None], lon=[None]))
        frame_data.append(go.Scattermapbox(lat=[None], lon=[None]))
        frame_data.append(go.Scattermapbox(
            lat=[bs.lat for bs in base_stations], lon=[bs.lon for bs in base_stations],
            customdata=_tn_radius_hover(base_stations),
            hovertemplate="%{customdata}<extra></extra>",
        ))
            
        frames.append(go.Frame(name=hour_str, data=frame_data, traces=[0, 1, 2, 3, 4, 5, 6]))
        slider_steps.append({"args": [[hour_str], {"frame": {"duration": 600, "redraw": True}, "mode": "immediate"}], "label": hour_str, "method": "animate"})

    fig.frames = frames

    mapbox_layers = []
    if visual_mode == "Geometric (Voronoi)" and SHOW_KMEANS_SHAPES:
        mapbox_layers.append(dict(source=voronoi_geojson, type="fill", color="rgba(0, 191, 255, 0.12)"))
        mapbox_layers.append(dict(source=voronoi_geojson, type="line", color="rgba(0, 191, 255, 0.55)", line=dict(width=1)))

    if SHOW_TN_CIRCLES:
        mapbox_layers.append(dict(source=coverage_geojson, type="fill", color="rgba(255, 165, 0, 0.06)"))
        mapbox_layers.append(dict(source=coverage_geojson, type="line", color="rgba(255, 165, 0, 0.85)", line=dict(width=2)))

    if ontario_yaml.get("geojson_geometry"):
        mapbox_layers.append(dict(source=ontario_yaml["geojson_geometry"], type="line", color="cyan", line=dict(width=2)))

    fig.update_layout(
        template="plotly_dark", height=800, showlegend=True,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
        mapbox=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=5, layers=mapbox_layers),
        margin={"r":0,"t":40,"l":0,"b":0},
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}], "label": "Play ▶", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}], "label": "Pause ⏸", "method": "animate"}
            ],
            "type": "buttons", "direction": "left", "pad": {"r": 10, "t": 87}, "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top", "showactive": False
        }],
        sliders=[{"active": 0, "steps": slider_steps, "x": 0.1, "y": 0, "len": 0.9, "xanchor": "left", "yanchor": "top", "pad": {"b": 10, "t": 50}}]
    )
    fig.write_html(filename)

# ==========================================
# CONFIGURATION MERGING
# ==========================================
base_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg_defaults, resolve=True))
if "defaults" in base_cfg: del base_cfg["defaults"]

cfg = OmegaConf.merge(base_cfg, constellation_cfg_defaults, {
    "scenario": scenario_yaml_cfg, "population": population_yaml_cfg,
    "terrestrial": terrestrial_yaml_cfg, "cost": cost_yaml_cfg,
    "mobility": mobility_yaml_cfg, "optimization": optimization_yaml_cfg,
})

dynamic_overrides = OmegaConf.create({
    "scenario": {
        "name": ontario_yaml.get("name", "Ontario"),
        "h3_resolution": int(ontario_yaml.get("h3_resolution", 3)),
        "geojson_geometry": ontario_yaml.get("geojson_geometry", {}),
    },
    # LEO controls use constellation.yaml as defaults and become live overrides
    # only when the user changes them in the sidebar.
    "constellation": {
        "altitude_km": float(SAT_ALTITUDE),
        "total_satellites": int(TOTAL_SATS),
        "bandwidth_hz": float(NTN_BW_MHZ * 1e6),
        "eirp_dbw": float(SAT_EIRP),
    },
    "population": {
        "total_city_users": int(TOTAL_USERS * CITY_RATIO),
        "total_rural_users": int(TOTAL_USERS - int(TOTAL_USERS * CITY_RATIO)),
        "traffic": {"diurnal_curve": {"evening_peak": {"center_hour": float(EVENING_PEAK_HOUR)}}},
    },
    "terrestrial": {
        "density_threshold": int(TN_CITY_THRESHOLD),
        "users_per_cluster_ratio": int(TN_USERS_PER_TOWER),
        "min_users_per_tn_cluster": int(TN_USERS_PER_TOWER),
        "coverage_radius_min_km": float(TN_RADIUS_MIN_KM),
        "coverage_radius_km": float(TN_RADIUS_MAX_KM),
        "coverage_radius_max_km": float(TN_RADIUS_MAX_KM),
        "max_extent_km": float(TN_RADIUS_MAX_KM),
        "bs_capacity_mbps": float(TN_BS_CAPACITY_MBPS),
        "bandwidth_hz": float(TN_BW_MHZ * 1e6),
        "p_tx_dbm": float(TN_P_TX),
        "g_tx_dbi": float(TN_G_TX),
        "g_rx_ue_dbi": float(TN_G_RX),
        "carrier_freq_hz": float(TN_FREQ_GHZ * 1e9),
        "sinr_min_db": float(TN_SINR_MIN),
        "shadowing_std_dev_db": float(TN_SHADOWING),
        "body_loss_db": float(TN_BODY_LOSS),
        "use_physical_radius": True,
        "fixed_coverage_radius_km": False,
    },
    "simulation": {
        "duration_s": int(SIM_DURATION),
        "time_step_s": int(TIME_STEP),
        "allow_spillover": False,
        "allow_degraded_service": False,
    },
})

cfg = OmegaConf.merge(cfg, dynamic_overrides)
cfg.simulation.allow_spillover = False
cfg.simulation.allow_degraded_service = False
selected_profiles = {"light": use_light, "medium": use_medium, "heavy": use_heavy}
yaml_profiles = OmegaConf.to_container(population_yaml_cfg.traffic.profiles, resolve=True)
active_profiles = {p: dict(yaml_profiles[p]) for p, enabled in selected_profiles.items() if enabled and p in yaml_profiles}
if not active_profiles:
    st.sidebar.warning("At least one traffic profile must be active. Medium users were enabled automatically.")
    active_profiles = {"medium": dict(yaml_profiles["medium"])}

prob_sum = sum(float(profile["probability"]) for profile in active_profiles.values())
for profile in active_profiles.values():
    profile["probability"] = float(profile["probability"]) / prob_sum
cfg.population.traffic.profiles = OmegaConf.create(active_profiles)

cfg_yaml_string = OmegaConf.to_yaml(cfg)

# ==========================================
# EXECUTION (WITH SESSION STATE CACHING)
# ==========================================
if "last_cfg" not in st.session_state or st.session_state.last_cfg != cfg_yaml_string:
    with st.spinner("Executing Digital Twin Physics Engine..."):
        active_region = Region(name=cfg.scenario.name, geojson_geometry=cfg.scenario.geojson_geometry, h3_resolution=cfg.scenario.h3_resolution)
        tessellate_region(active_region, pad_edges=True)
        
        total_sats = int(cfg.constellation.get("total_satellites", 1584))
        num_planes = int(cfg.constellation.get("num_planes", max(1, int(total_sats / 18))))
        walker = WalkerParameters(
            total_sats,
            num_planes,
            int(cfg.constellation.get("phasing", 1)),
            float(cfg.constellation.get("inclination_deg", 53.0)),
            float(cfg.constellation.get("altitude_km", 550.0)),
            OrbitType.LEO,
        )
        leo = LEOConstellation(params=walker, name=cfg.constellation.name)
        
        users = generate_users(cfg, active_region)
        towers = generate_terrestrial_network(cfg, users, active_region.h3_resolution)
        beam_data, user_data = run_daily_mobility_simulation(cfg, users, towers, leo, active_region)
        
        # Save to cache
        st.session_state.physics_data = (active_region, users, towers, beam_data, user_data)
        st.session_state.last_cfg = cfg_yaml_string

# Always ensure map reflects current visual mode
visual_state_key = f"{cfg_yaml_string}_{VISUAL_MODE}_{SHOW_KMEANS_SHAPES}_{SHOW_TN_CIRCLES}_{SHOW_LEO_HEXAGONS}"
html_filename = "Final_Animation.html"

if "last_visual_state" not in st.session_state or st.session_state.last_visual_state != visual_state_key:
    with st.spinner("Rendering Visualization..."):
        active_region, users, towers, beam_data, user_data = st.session_state.physics_data
        render_custom_dashboard_animation(active_region, users, towers, beam_data, user_data, cfg.simulation.duration_s, cfg.simulation.time_step_s, html_filename, VISUAL_MODE)
        st.session_state.last_visual_state = visual_state_key

st.success(f"Simulation complete. Running in {VISUAL_MODE} mode ")

# ==========================================
# DISPLAY DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Interactive Simulation Map", "STEPS Mobility", "Traffic Analytics", "Network Utilization", "Data Exports"])

df_users_anim = pd.read_csv("user_hourly_states.csv") if os.path.exists("user_hourly_states.csv") else pd.DataFrame()
df_summary = pd.read_csv("system_summary_table.csv") if os.path.exists("system_summary_table.csv") else pd.DataFrame()
df_usage = pd.read_csv("network_usage_data.csv") if os.path.exists("network_usage_data.csv") else pd.DataFrame()

with tab1:
    st.subheader(f" TN/LEO runtime association - {VISUAL_MODE}")
    if os.path.exists(html_filename):
        with open(html_filename, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=850)

with tab2:
    if not df_users_anim.empty:
        bx, by = get_boundary_coords(ONTARIO_GEOM)
        colA, colB = st.columns(2)
        with colA:
            fig2A = go.Figure()
            fig2A.add_trace(go.Scattermapbox(lat=by, lon=bx, mode='lines', line=dict(color='white', width=1.0), showlegend=False))
            tracked_users = random.sample(list(df_users_anim['User_ID'].unique()), min(8, len(df_users_anim['User_ID'].unique())))
            for uid in tracked_users:
                user_path = df_users_anim[df_users_anim['User_ID'] == uid]
                fig2A.add_trace(go.Scattermapbox(lat=user_path['Lat'], lon=user_path['Lon'], mode='lines+markers', name=f"User {uid}"))
            fig2A.update_layout(template="plotly_dark", mapbox=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=4.2), height=550, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2A, use_container_width=True)
        with colB:
            fig2B = go.Figure()
            fig2B.add_trace(go.Scattermapbox(lat=by, lon=bx, mode='lines', line=dict(color='white', width=1.0), showlegend=False))
            fig2B.add_trace(go.Scattermapbox(lat=df_users_anim['Lat'], lon=df_users_anim['Lon'], mode='markers', marker=dict(color='cyan', size=3, opacity=0.03), showlegend=False))
            fig2B.update_layout(template="plotly_dark", mapbox=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=4.2), height=550, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2B, use_container_width=True)

with tab3:
    if not df_summary.empty:
        df_summary['Continuous_Hour'] = df_summary['Time_s'] / 3600.0
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Total_Demand_Mbps'], mode='lines', name='Total Demand (Mbps)', line=dict(color='cyan', width=3)))
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Served_TN_Mbps'], mode='lines', name='Served by TN (5G)', line=dict(color='deepskyblue', width=2, dash='dash')))
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Served_NTN_Mbps'], mode='lines', name='Served by LEO (Sat)', line=dict(color='green', width=2, dash='dot')))
        fig3.add_trace(go.Scatter(x=df_summary['Continuous_Hour'], y=df_summary['Dropped_Traffic_Mbps'], mode='lines', name='Dropped Traffic (Outage)', line=dict(color='red', width=2)))
        fig3.update_layout(xaxis_title="Time of Day (Hours)", yaxis_title="Data Load (Mbps)", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig3.update_xaxes(tickvals=list(range(0, 25, 2)), gridcolor='rgba(0, 0, 0, 0.1)')
        fig3.update_yaxes(gridcolor='rgba(0, 0, 0, 0.1)')
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    if not df_usage.empty:
        st.subheader("Physical Hardware Utilization")
        usage_summary = df_usage.groupby(['Hour', 'Network_Type'])['Utilization_%'].mean().reset_index()
        usage_summary['Time'] = usage_summary['Hour'].str.extract(r'(\d+\.\d+)').astype(float)
        usage_summary = usage_summary.sort_values(by=["Network_Type", "Time"])
        fig4 = px.line(usage_summary, x="Time", y="Utilization_%", color="Network_Type", color_discrete_map={"5G_TN": "orange", "LEO_NTN": "green"})
        fig4.update_layout(xaxis_title="Time of Day (Hours)", yaxis_title="Bandwidth Used (%)", template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig4.update_yaxes(range=[0, 105], gridcolor='rgba(0, 0, 0, 0.1)')
        fig4.update_xaxes(tickvals=list(range(0, 25, 2)), gridcolor='rgba(0, 0, 0, 0.1)')
        st.plotly_chart(fig4, use_container_width=True)

with tab5:
    st.subheader("Data Export & Previews")
    c1, c2, c3 = st.columns(3)
    files = [("users_initial_state.csv", c1), ("user_hourly_states.csv", c1), ("system_summary_table.csv", c2), ("network_usage_data.csv", c2), ("detailed_drop_log.csv", c3)]
    for f_name, col in files:
        if os.path.exists(f_name):
            with col:
                st.markdown(f"**{f_name}**")
                st.dataframe(pd.read_csv(f_name).head(10), height=200)
                st.download_button(f"Download {f_name}", data=open(f_name, "rb"), file_name=f_name)