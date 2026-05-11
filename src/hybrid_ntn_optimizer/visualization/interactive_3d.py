import numpy as np
import plotly.graph_objects as go
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation

def plot_3d_interactive_globe(dt_s: float = 0.0):
    print("Generating Starlink Phase 1 Constellation for 3D...")
    leo = LEOConstellation.starlink_shell1()
    states = leo.snapshot(dt_s=dt_s)
    
    # 1. Extract the ECI 3D coordinates (convert meters to kilometers)
    x_vals = [state.position_eci.x / 1000.0 for state in states]
    y_vals = [state.position_eci.y / 1000.0 for state in states]
    z_vals = [state.position_eci.z / 1000.0 for state in states]
    
    # 2. Build the Earth Sphere using NumPy math
    R_earth = 6371.0  # Earth's mean radius in km
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = R_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 3. Create the Plotly Figure
    fig = go.Figure()
    
    # 4. Add the Earth to the center of the scene
    fig.add_surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues',       # Makes it look like a blue marble
        showscale=False,          # Hide the color bar
        opacity=0.9,
        hoverinfo='skip',         # Don't show text when hovering over the Earth
        name='Earth'
    )
    
    # 5. Add the 1,584 Satellites!
    fig.add_scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.8
        ),
        name=f'{leo.name} ({leo.num_satellites} sats)',
        hoverinfo='text',
        text=[s.satellite_id for s in states] # Hover over a dot to see its ID!
    )
    
    # 6. Configure the 3D Scene layout
    fig.update_layout(
        title="Interactive 3D Walker-Delta Constellation",
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data', # CRITICAL: Ensures the Earth is perfectly round, not stretched
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template='plotly_dark' # Space is dark!
    )
    
    print("Saving interactive 3D map to HTML...")
    # Save it as a standalone webpage
    fig.write_html("interactive_constellation.html")
    print("Done! Open 'interactive_constellation.html' in your Windows web browser.")

if __name__ == "__main__":
    plot_3d_interactive_globe(dt_s=0.0)