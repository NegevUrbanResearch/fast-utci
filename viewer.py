"""
3D Model Viewer for fast-utci

This script provides a 3D viewer to visualize loaded 3D models and validate
that the reader.py module is working correctly.
"""

import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
# Import will be done in main() function


class ModelViewer:
    """Class for visualizing 3D models and weather data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def visualize_model_3d(self, mesh, title: str = "3D Model", 
                          show_normals: bool = False, 
                          show_vertices: bool = False) -> go.Figure:
        """
        Create a 3D visualization of the model using plotly.
        
        Args:
            mesh: trimesh object
            title: Title for the plot
            show_normals: Whether to show face normals
            show_vertices: Whether to show vertices
            
        Returns:
            plotly.graph_objects.Figure: 3D plot
        """
        # Create mesh visualization
        fig = go.Figure()
        
        # Add the mesh surface
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1], 
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.7,
            colorscale='Viridis',
            name='Mesh Surface'
        ))
        
        # Optionally show vertices
        if show_vertices:
            fig.add_trace(go.Scatter3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                mode='markers',
                marker=dict(size=2, color='red'),
                name='Vertices'
            ))
        
        # Optionally show face normals
        if show_normals:
            # Calculate face centers and normals
            face_centers = np.mean(mesh.vertices[mesh.faces], axis=1)
            face_normals = mesh.face_normals
            
            # Create normal vectors (scaled down for visibility)
            normal_scale = 0.1
            normal_end_points = face_centers + face_normals * normal_scale
            
            # Add normal vectors
            for i in range(min(100, len(face_centers))):  # Limit to 100 normals for performance
                fig.add_trace(go.Scatter3d(
                    x=[face_centers[i, 0], normal_end_points[i, 0]],
                    y=[face_centers[i, 1], normal_end_points[i, 1]],
                    z=[face_centers[i, 2], normal_end_points[i, 2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_weather_data(self, weather_df: pd.DataFrame) -> go.Figure:
        """
        Create visualizations of weather data.
        
        Args:
            weather_df: DataFrame with weather data
            
        Returns:
            plotly.graph_objects.Figure: Weather plots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Air Temperature', 'Wind Speed', 
                          'Relative Humidity', 'Solar Radiation'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot air temperature
        fig.add_trace(
            go.Scatter(x=weather_df['datetime'], y=weather_df['air_temp'],
                      mode='lines', name='Air Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        # Plot wind speed
        fig.add_trace(
            go.Scatter(x=weather_df['datetime'], y=weather_df['wind_speed'],
                      mode='lines', name='Wind Speed', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Plot relative humidity
        fig.add_trace(
            go.Scatter(x=weather_df['datetime'], y=weather_df['relative_humidity'],
                      mode='lines', name='Relative Humidity', line=dict(color='green')),
            row=2, col=1
        )
        
        # Plot solar radiation
        fig.add_trace(
            go.Scatter(x=weather_df['datetime'], y=weather_df['global_horizontal_radiation'],
                      mode='lines', name='Solar Radiation', line=dict(color='orange')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Weather Data Overview',
            height=600,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date/Time")
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="Humidity (%)", row=2, col=1)
        fig.update_yaxes(title_text="Radiation (W/m²)", row=2, col=2)
        
        return fig
    
    def visualize_model_and_stats(self, mesh, title: str = "3D Model & Statistics") -> go.Figure:
        """
        Create a clean visualization with 3D model (75%) and statistics table (25%).
        
        Args:
            mesh: trimesh object
            title: Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Model and stats plot
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('3D Model', 'Model Statistics'),
            specs=[[{'type': 'scene'}, {'type': 'table'}]],
            column_widths=[0.75, 0.25],  # 75% for 3D model, 25% for table
            horizontal_spacing=0.05
        )
        
        # Add the 3D mesh
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1], 
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            opacity=0.7,
            colorscale='Viridis',
            name='Mesh Surface'
        ), row=1, col=1)
        
        # Add vertices as red points
        fig.add_trace(go.Scatter3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            mode='markers',
            marker=dict(size=2, color='red'),
            name='Vertices',
            showlegend=False
        ), row=1, col=1)
        
        # Create model statistics table
        model_info = {
            'Property': ['Vertices', 'Faces', 'Surface Area (m²)', 'Scale'],
            'Value': [
                f"{len(mesh.vertices):,}",
                f"{len(mesh.faces):,}",
                f"{mesh.area:.2f}",
                f"{mesh.scale:.4f}"
            ]
        }
        
        fig.add_trace(go.Table(
            header=dict(values=list(model_info.keys()),
                       fill_color='lightblue',
                       align='center',
                       font=dict(size=14, color='black')),
            cells=dict(values=list(model_info.values()),
                      fill_color='white',
                      align='center',
                      font=dict(size=12, color='black'))
        ), row=1, col=2)
        
        # Update 3D scene
        fig.update_scenes(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            row=1, col=1
        )
        
        fig.update_layout(
            title=title,
            height=800,
            width=2400,
            showlegend=False
        )
        
        return fig
    
    def visualize_weather_data_clean(self, weather_df: pd.DataFrame, 
                                   title: str = "Weather Data") -> go.Figure:
        """
        Create a clean 2x2 grid visualization of weather data.
        
        Args:
            weather_df: DataFrame with weather data
            title: Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Weather data plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Air Temperature', 'Wind Speed',
                          'Relative Humidity', 'Solar Radiation'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add weather data traces
        # Air temperature
        fig.add_trace(go.Scatter(
            x=weather_df['datetime'], 
            y=weather_df['air_temp'],
            mode='lines', 
            name='Air Temperature', 
            line=dict(color='red', width=2),
            showlegend=False
        ), row=1, col=1)
        
        # Wind speed
        fig.add_trace(go.Scatter(
            x=weather_df['datetime'], 
            y=weather_df['wind_speed'],
            mode='lines', 
            name='Wind Speed', 
            line=dict(color='blue', width=2),
            showlegend=False
        ), row=1, col=2)
        
        # Relative humidity
        fig.add_trace(go.Scatter(
            x=weather_df['datetime'], 
            y=weather_df['relative_humidity'],
            mode='lines', 
            name='Relative Humidity', 
            line=dict(color='green', width=2),
            showlegend=False
        ), row=2, col=1)
        
        # Solar radiation
        fig.add_trace(go.Scatter(
            x=weather_df['datetime'], 
            y=weather_df['global_horizontal_radiation'],
            mode='lines', 
            name='Solar Radiation', 
            line=dict(color='orange', width=2),
            showlegend=False
        ), row=2, col=2)
        
        # Update weather data axes
        fig.update_xaxes(title_text="Date/Time", row=1, col=1)
        fig.update_xaxes(title_text="Date/Time", row=1, col=2)
        fig.update_xaxes(title_text="Date/Time", row=2, col=1)
        fig.update_xaxes(title_text="Date/Time", row=2, col=2)
        
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Wind Speed (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="Humidity (%)", row=2, col=1)
        fig.update_yaxes(title_text="Radiation (W/m²)", row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            width=2400,
            showlegend=False
        )
        
        return fig
    
    def show_model_info(self, model_info: dict, weather_info: dict):
        """
        Display model and weather information in a formatted way.
        
        Args:
            model_info: Dictionary with model information
            weather_info: Dictionary with weather information
        """
        print("=" * 50)
        print("3D MODEL INFORMATION")
        print("=" * 50)
        print(f"Vertices: {model_info['vertices_count']:,}")
        print(f"Faces: {model_info['faces_count']:,}")
        print(f"Volume: {model_info['volume']:.2f} m³" if model_info['volume'] else "Volume: Not available")
        print(f"Surface Area: {model_info['surface_area']:.2f} m²")
        print(f"Scale: {model_info['scale']:.4f}")
        print(f"Watertight: {model_info['is_watertight']}")
        print(f"Winding Consistent: {model_info['is_winding_consistent']}")
        
        bounds = model_info['bounds']
        print(f"Bounds: X[{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] "
              f"Y[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] "
              f"Z[{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
        
        center = model_info['center']
        print(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        print("\n" + "=" * 50)
        print("WEATHER DATA INFORMATION")
        print("=" * 50)
        loc = weather_info['location']
        print(f"Location: {loc['city']}, {loc['country']}")
        print(f"Coordinates: {loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E")
        print(f"Elevation: {loc['elevation']:.1f} m")
        print(f"Timezone: UTC{loc['timezone']:+g}")
        print(f"Data Points: {weather_info['data_points']:,} hours")
        print(f"Date Range: {weather_info['date_range']['start']} to {weather_info['date_range']['end']}")


def main():
    """Main function to demonstrate the viewer."""
    # Set up minimal logging
    logging.basicConfig(level=logging.WARNING)
    
    # Initialize viewer
    viewer = ModelViewer()
    
    # File paths
    model_path = "data/rec_model_no_curve.glb"
    weather_path = "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    
    try:
        print("Loading project data...")
        
        # Read the data
        from reader import read_project_data, validate_project_data, ModelReader, WeatherReader
        model, weather = read_project_data(model_path, weather_path)
        
        # Get information
        model_reader = ModelReader()
        weather_reader = WeatherReader()
        model_info = model_reader.get_model_info(model)
        weather_info = weather_reader.get_weather_info(weather)
        
        # Display information
        viewer.show_model_info(model_info, weather_info)
        
        # Extract weather data for visualization
        weather_df = weather_reader.extract_utci_inputs(weather)
        
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)
        
        # Create separate clean visualizations
        print("Opening 3D Model & Statistics visualization...")
        model_fig = viewer.visualize_model_and_stats(
            model, 
            title="3D Model & Statistics - fast-utci"
        )
        model_fig.show()
        
        print("Opening Weather Data visualization...")
        weather_fig = viewer.visualize_weather_data_clean(
            weather_df,
            title="Weather Data - fast-utci"
        )
        weather_fig.show()
        
        # Validate the data
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)
        
        is_valid, validation_results = validate_project_data(model, weather)
        
        if is_valid:
            print("✅ All validation checks PASSED!")
        else:
            print("❌ Some validation checks FAILED:")
            for data_type, issues in validation_results.items():
                if issues:
                    print(f"  {data_type.upper()}:")
                    for issue in issues:
                        print(f"    - {issue}")
        
        print("\n" + "=" * 50)
        print("SUCCESS: Reader module is working correctly!")
        print("=" * 50)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check that the data files exist in the correct locations.")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
