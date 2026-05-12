import hydra
from omegaconf import DictConfig
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.visualization.maps import plot_global_constellation,plot_2d_interactive_animation
from hybrid_ntn_optimizer.visualization.interactive_3d import plot_3d_animated_globe
from hybrid_ntn_optimizer.visualization.coverage import plot_hex_coverage_animation


@hydra.main(version_base=None, config_path="configs", config_name="base")
def run_simulation(cfg: DictConfig):
    print(f"=== Running Propagation & Visibility Test ===")
    
    # 1. Build Constellation
    leo = LEOConstellation.from_dict(cfg.constellation, epoch_utc=cfg.epoch_utc)
    active_region = Region(
        name=cfg.scenario.name,
        geojson_geometry=cfg.scenario.geojson_geometry,
        h3_resolution=cfg.scenario.h3_resolution
    )
    tessellate_region(active_region)

    # 2. Simulation Loop
    # We will step through time to see satellites moving
    steps = int(cfg.sim_duration_s / cfg.time_step_s)
    
    print(f"\nSimulating {cfg.sim_duration_s}s at {cfg.observer.name} ({cfg.observer.lat}, {cfg.observer.lon})")
    print(f"{'Time (s)':<10} | {'Visible':<8} | {'Best Satellite':<20} | {'Elev (deg)':<10}")
    print("-" * 60)

    for step in range(steps + 1):
        dt_s = step * cfg.time_step_s
        
        # Check visibility for the observer
        visible_sats = leo.visible_from(
            lat_deg=cfg.observer.lat, 
            lon_deg=cfg.observer.lon, 
            dt_s=dt_s
        )
        
        # Get the highest satellite for the link budget
        best = leo.best_satellite_from(
            lat_deg=cfg.observer.lat, 
            lon_deg=cfg.observer.lon, 
            dt_s=dt_s
        )
        
        best_id = best.satellite_id if best else "None"
        best_el = f"{best.elevation_deg:0.2f}" if best else "N/A"
        
        print(f"{dt_s:<10} | {len(visible_sats):<8} | {best_id:<20} | {best_el:<10}")

    # 3. Final Visualization (Snapshot at t=0)
    if cfg.visualize_2d:
        plot_global_constellation(leo, dt_s=0.0, save_path="constellation_start.png")

    if cfg.visualize_2d:
        plot_2d_interactive_animation(leo, cfg.sim_duration_s, cfg.time_step_s)
        
    if cfg.visualize_3d:
        plot_3d_animated_globe(leo, cfg.sim_duration_s, cfg.time_step_s)
    plot_hex_coverage_animation(
        leo=leo, 
        region=active_region, 
        duration_s=cfg.sim_duration_s, 
        time_step_s=cfg.time_step_s
    )
    print("\nPropagation test complete. Files generated.")

if __name__ == "__main__":
    run_simulation()