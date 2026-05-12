import hydra
from omegaconf import DictConfig
from hybrid_ntn_optimizer.models.scenario import Region

# Import the grid builder (ensure you are using the one we updated with edge padding!)
from hybrid_ntn_optimizer.coverage.mapper   import tessellate_region

# Import your new fast-plot function
from hybrid_ntn_optimizer.visualization.coverage import plot_static_grid

@hydra.main(version_base=None, config_path="configs", config_name="base")
def test_grid(cfg: DictConfig):
    print("=== Fast Grid Test ===")
    
    # 1. Load the Region from your new ontario_full.yaml
    active_region = Region(
        name=cfg.scenario.name, 
        geojson_geometry=cfg.scenario.geojson_geometry, 
        h3_resolution=cfg.scenario.h3_resolution
    )
    
    # 2. Tessellate (Run the H3 math with padding)
    tessellate_region(active_region, pad_edges=True)
    
    # 3. Plot statically! No simulation loop.
    plot_static_grid(active_region, filename="fast_grid_test.html")

if __name__ == "__main__":
    test_grid()