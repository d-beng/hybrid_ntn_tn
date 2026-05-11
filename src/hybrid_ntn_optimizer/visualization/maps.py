import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation

def plot_global_constellation(dt_s: float = 0.0):
    """
    Plots the instantaneous ground tracks of the LEO constellation on a world map.
    """
    print("Generating Starlink Phase 1 Constellation...")
    # 1. Initialize our verified physics engine
    leo = LEOConstellation.starlink_shell1()
    
    # 2. Take a snapshot of the fleet at a specific second in time
    print(f"Calculating physics and propagation for {leo.num_satellites} satellites...")
    states = leo.snapshot(dt_s=dt_s)
    
    # 3. Extract the latitudes and longitudes
    lats = [state.lat_deg for state in states]
    lons = [state.lon_deg for state in states]
    
    # 4. Set up the Matplotlib figure and the Cartopy map projection
    fig = plt.figure(figsize=(15, 8))
    
    # PlateCarree is the standard flat 2D map projection
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add map features (coastlines, borders, oceans)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    
    # Add gridlines (lat/lon lines)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 5. Plot the satellites as scatter points!
    print("Drawing map...")
    ax.scatter(
        lons, 
        lats, 
        color='red', 
        s=5,           # 's' is the size of the dot
        marker='o', 
        transform=ccrs.PlateCarree(), 
        label=f"{leo.name} ({leo.num_satellites} sats)"
    )
    
    # 6. Add titles and legends
    plt.title(f"Global Satellite Constellation Coverage (t = {dt_s}s)", fontsize=16, pad=20)
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig("starlink_constellation_map.png", dpi=300, bbox_inches='tight')
    print("Map successfully saved as 'starlink_constellation_map.png'!")

if __name__ == "__main__":
    # Test the visualizer when running this script directly
    plot_global_constellation(dt_s=0.0)