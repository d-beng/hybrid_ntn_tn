import pandas as pd
import hydra
from omegaconf import DictConfig

# 1. Models & Core Types
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.core.types import WalkerParameters, OrbitType
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation

# 2. Generators & Coverage Mapping
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.traffic.profiles import generate_users
from hybrid_ntn_optimizer.terrestrial.coverage import generate_terrestrial_network

# 3. Simulation Engine
from hybrid_ntn_optimizer.simulation.full_pipeline import run_daily_mobility_simulation
from hybrid_ntn_optimizer.visualization.plots import plot_master_hybrid_animation

@hydra.main(version_base=None, config_path="configs", config_name="base")
def run_simulation(cfg: DictConfig):
    print("\n" + "="*50)
    print("🚀 INITIALIZING HYBRID NTN-TN SIMULATOR")
    print("="*50)
    
    # ==========================================
    # PHASE 1: THE GEOGRAPHY
    # ==========================================
    print("\n[Phase 1] Building Geographic Map...")
    active_region = Region(
        name=cfg.scenario.name, 
        geojson_geometry=cfg.scenario.geojson_geometry, 
        h3_resolution=cfg.scenario.h3_resolution
    )
    # Fill the region with hexagons (including edge padding)
    tessellate_region(active_region, pad_edges=True)
    
    # ==========================================
    # PHASE 2: THE SPACE SEGMENT
    # ==========================================
    print("\n[Phase 2] Generating Space Segment (LEO Constellation)...")
    
    # Dynamically read orbital mechanics from YAML
    walker_params = WalkerParameters(
        total_satellites=cfg.constellation.total_satellites,
        num_planes=cfg.constellation.num_planes,
        phasing=cfg.constellation.phasing,
        inclination_deg=cfg.constellation.inclination_deg,
        altitude_km=cfg.constellation.altitude_km,
        orbit_type=OrbitType.LEO
    )
    
    # Launch the constellation with hardware parameters
    leo = LEOConstellation(
        params=walker_params,
        name=cfg.constellation.get("name", "Starlink-Shell-1"),
        eirp_dbw=cfg.constellation.get("eirp_dbw", 40.0),
        g_t_db=cfg.constellation.get("g_t_db", 10.0),
        max_spot_beams=cfg.constellation.get("max_spot_beams", 15),
        beam_radius_nadir_km=cfg.constellation.get("beam_radius_nadir_km", 200.0),
        max_steering_angle_deg=cfg.constellation.get("max_steering_angle_deg", 45.0)
    )
    print(f"✅ Deployed {leo.num_satellites} satellites into {walker_params.num_planes} orbital planes.")
    
    # ==========================================
    # PHASE 3: THE GROUND SEGMENT
    # ==========================================
    print("\n[Phase 3] Populating Ground Segment...")
    
    # 1. Spawn Users
    users = generate_users(cfg, active_region)
    print(f"✅ Generated {len(users)} Mobile Users.")
    
    # 2. Build 5G Towers using KMeans
    towers = generate_terrestrial_network(cfg, users, active_region.h3_resolution)
    
    # ==========================================
    # PHASE 4: THE HYBRID SIMULATION LOOP
    # ==========================================
    print("\n[Phase 4] Initiating 24-Hour Mobility & Traffic Engine...")
    
    # Pass all our generated objects into the simulation engine
    beam_animation_data,user_animation_data = run_daily_mobility_simulation(
        cfg=cfg, 
        users=users, 
        base_stations=towers, 
        leo=leo, 
        region=active_region
    )
    #print("📊 Loading sampled animation data from disk for visualization...")
    
    # Read the memory-safe 1% sample we flushed to the hard drive
    #user_animation_df = pd.read_csv("user_hourly_states.csv")
    
    # Convert it back into the list-of-dictionaries format your plotter expects
    #user_animation_data = user_animation_df.to_dict('records')

    
    # ==========================================
    # PHASE 5: THE MASTER VISUALIZATION
    # ==========================================
    plot_master_hybrid_animation(
        region=active_region, 
        users=users, 
        base_stations=towers, 
        beam_data=beam_animation_data, 
        user_data=user_animation_data,
        duration_s=cfg.simulation.duration_s, 
        time_step_s=cfg.simulation.time_step_s,
        filename="Final_Animation.html"
    )
    
    print("\n🎉 SIMULATION COMPLETE. Check output directories for CSV exports.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_simulation()