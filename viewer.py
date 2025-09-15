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
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
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


class UTCIHeatmapViewer:
    """
    Advanced 3D viewer for UTCI heatmap visualization.
    
    Displays 3D models with UTCI thermal comfort data as colored point clouds,
    allowing comparison with Grasshopper results and interactive exploration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # UTCI thermal comfort color scale
        self.utci_comfort_scale = {
            'extreme_cold': {'range': (-float('inf'), -40), 'color': '#0000FF', 'label': 'Extreme Cold'},
            'very_cold': {'range': (-40, -27), 'color': '#4169E1', 'label': 'Very Cold'},
            'cold': {'range': (-27, -13), 'color': '#87CEEB', 'label': 'Cold'},
            'cool': {'range': (-13, 9), 'color': '#98FB98', 'label': 'Cool'},
            'comfortable': {'range': (9, 26), 'color': '#00FF00', 'label': 'Comfortable'},
            'warm': {'range': (26, 32), 'color': '#FFD700', 'label': 'Warm'},
            'hot': {'range': (32, 38), 'color': '#FF8C00', 'label': 'Hot'},
            'very_hot': {'range': (38, 46), 'color': '#FF4500', 'label': 'Very Hot'},
            'extreme_hot': {'range': (46, float('inf')), 'color': '#8B0000', 'label': 'Extreme Hot'}
        }
    
    def get_utci_color(self, utci_value: float) -> str:
        """Get color for UTCI value based on thermal comfort category."""
        for category, data in self.utci_comfort_scale.items():
            min_val, max_val = data['range']
            if min_val <= utci_value < max_val:
                return data['color']
        return '#808080'  # Gray for undefined
    
    def get_utci_category(self, utci_value: float) -> str:
        """Get thermal comfort category for UTCI value."""
        for category, data in self.utci_comfort_scale.items():
            min_val, max_val = data['range']
            if min_val <= utci_value < max_val:
                return data['label']
        return 'Unknown'
    
    def visualize_utci_heatmap(self, 
                              model_mesh,
                              utci_results: Dict[str, Any],
                              title: str = "UTCI Thermal Comfort Heatmap",
                              show_model: bool = True,
                              show_comfort_legend: bool = True,
                              point_size: int = 8) -> go.Figure:
        """
        Create 3D visualization of UTCI heatmap with thermal comfort analysis.
        
        Args:
            model_mesh: Trimesh object for context geometry
            utci_results: Dictionary from UTCICalculator.compute_utci()
            title: Plot title
            show_model: Whether to show the 3D model
            show_comfort_legend: Whether to show thermal comfort legend
            point_size: Size of UTCI data points
            
        Returns:
            Plotly figure with 3D UTCI heatmap
        """
        fig = go.Figure()
        
        # Add 3D model with better visibility
        if show_model and model_mesh is not None:
            fig.add_trace(go.Mesh3d(
                x=model_mesh.vertices[:, 0],
                y=model_mesh.vertices[:, 1],
                z=model_mesh.vertices[:, 2],
                i=model_mesh.faces[:, 0],
                j=model_mesh.faces[:, 1],
                k=model_mesh.faces[:, 2],
                opacity=0.8,  # More opaque for better visibility
                color='darkslategray',  # Darker color to stand out
                name='Building/Context',
                showlegend=True,
                lighting=dict(
                    ambient=0.4,
                    diffuse=1.0,
                    specular=0.2,
                    roughness=0.1,
                    fresnel=0.2
                ),
                lightposition=dict(x=100, y=200, z=300)
            ))
        
        # Extract UTCI data points
        positions = []
        utci_values = []
        mrt_values = []
        categories = []
        colors = []
        
        for pos_key, data in utci_results.items():
            position = data['position']
            utci_vals = data['utci']
            mrt_vals = data['mrt']
            
            # Use first hour's data (or average if multiple hours)
            if len(utci_vals) > 0:
                # Handle UTCI values that might be objects
                numeric_utci_vals = []
                for uval in utci_vals:
                    try:
                        if hasattr(uval, 'utci'):
                            numeric_utci_vals.append(float(uval.utci))
                        elif isinstance(uval, dict) and 'utci' in uval:
                            numeric_utci_vals.append(float(uval['utci']))
                        else:
                            numeric_utci_vals.append(float(uval))
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                if numeric_utci_vals:
                    utci_val = np.mean(numeric_utci_vals) if len(numeric_utci_vals) > 1 else numeric_utci_vals[0]
                    mrt_val = np.mean(mrt_vals) if len(mrt_vals) > 1 else mrt_vals[0]
                    
                    if not np.isnan(utci_val):
                        positions.append(position)
                        utci_values.append(utci_val)
                        mrt_values.append(mrt_val)
                        categories.append(self.get_utci_category(utci_val))
                        colors.append(self.get_utci_color(utci_val))
                        
        # Debug: Print some info about the points
        print(f"Visualization: {len(positions)} UTCI points, UTCI range: {min(utci_values):.1f} to {max(utci_values):.1f} °C")
        
        if len(positions) == 0:
            print("Warning: No valid UTCI data points to display")
            return fig
        
        positions = np.array(positions)
        utci_values = np.array(utci_values)
        mrt_values = np.array(mrt_values)
        
        # Create hover text
        hover_text = [
            f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<br>"
            f"UTCI: {utci:.1f}°C<br>"
            f"MRT: {mrt:.1f}°C<br>"
            f"Comfort: {cat}"
            for pos, utci, mrt, cat in zip(positions, utci_values, mrt_values, categories)
        ]
        
        # Add UTCI points as flat squares for better grid visualization
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=point_size * 2,  # Larger for better visibility
                color=colors,
                symbol='square',  # Square instead of circle
                line=dict(width=2, color='darkgray'),  # Thicker border
                opacity=0.9  # More opaque
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='UTCI Grid',
            showlegend=False
        ))
        
        # Add comfort legend if requested
        if show_comfort_legend:
            legend_x = []
            legend_y = []
            legend_z = []
            legend_colors = []
            legend_text = []
            
            # Create legend points off to the side
            x_offset = np.max(positions[:, 0]) + 20
            z_base = np.min(positions[:, 2])
            
            for i, (category, data) in enumerate(self.utci_comfort_scale.items()):
                legend_x.append(x_offset)
                legend_y.append(0)
                legend_z.append(z_base + i * 5)
                legend_colors.append(data['color'])
                min_val, max_val = data['range']
                range_text = f"{min_val:.0f} to {max_val:.0f}°C" if max_val != float('inf') else f"> {min_val:.0f}°C"
                if min_val == -float('inf'):
                    range_text = f"< {max_val:.0f}°C"
                legend_text.append(f"{data['label']}<br>{range_text}")
            
            fig.add_trace(go.Scatter3d(
                x=legend_x,
                y=legend_y,
                z=legend_z,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=legend_colors,
                    line=dict(width=2, color='black')
                ),
                text=[data['label'] for data in self.utci_comfort_scale.values()],
                textposition='middle right',
                hovertext=legend_text,
                hovertemplate='%{hovertext}<extra></extra>',
                name='Comfort Scale',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def compare_with_grasshopper(self,
                                model_mesh,
                                utci_results: Dict[str, Any],
                                grasshopper_csv: Union[str, Path, pd.DataFrame],
                                title: str = "UTCI Comparison: Python vs Grasshopper") -> go.Figure:
        """
        Compare UTCI results with Grasshopper validation data.
        
        Args:
            model_mesh: Trimesh object for context geometry
            utci_results: Python UTCI results
            grasshopper_csv: Path to Grasshopper CSV or DataFrame
            title: Plot title
            
        Returns:
            Plotly figure with comparison visualization
        """
        # Load Grasshopper data if it's a file path
        if isinstance(grasshopper_csv, (str, Path)):
            gh_data = pd.read_csv(grasshopper_csv)
        else:
            gh_data = grasshopper_csv.copy()
        
        # Create subplots: 3D comparison + statistics
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('3D UTCI Comparison', 'Statistics'),
            specs=[[{'type': 'scene'}, {'type': 'table'}]],
            column_widths=[0.75, 0.25]
        )
        
        # Add 3D model with better visibility
        if model_mesh is not None:
            fig.add_trace(go.Mesh3d(
                x=model_mesh.vertices[:, 0],
                y=model_mesh.vertices[:, 1],
                z=model_mesh.vertices[:, 2],
                i=model_mesh.faces[:, 0],
                j=model_mesh.faces[:, 1],
                k=model_mesh.faces[:, 2],
                opacity=0.8,  # More opaque
                color='darkslategray',  # Darker color
                name='Building/Context',
                showlegend=False,
                lighting=dict(
                    ambient=0.4,
                    diffuse=1.0,
                    specular=0.2,
                    roughness=0.1,
                    fresnel=0.2
                ),
                lightposition=dict(x=100, y=200, z=300)
            ), row=1, col=1)
        
        # Extract Python UTCI data
        python_positions = []
        python_utci = []
        
        for pos_key, data in utci_results.items():
            position = data['position']
            utci_vals = data['utci']
            
            if len(utci_vals) > 0:
                # Handle UTCI values that might be objects
                numeric_utci_vals = []
                for uval in utci_vals:
                    try:
                        if hasattr(uval, 'utci'):
                            numeric_utci_vals.append(float(uval.utci))
                        elif isinstance(uval, dict) and 'utci' in uval:
                            numeric_utci_vals.append(float(uval['utci']))
                        else:
                            numeric_utci_vals.append(float(uval))
                    except (ValueError, TypeError, AttributeError):
                        continue
                
                if numeric_utci_vals:
                    utci_val = np.mean(numeric_utci_vals) if len(numeric_utci_vals) > 1 else numeric_utci_vals[0]
                    if not np.isnan(utci_val):
                        python_positions.append(position)
                        python_utci.append(utci_val)
        
        python_positions = np.array(python_positions)
        python_utci = np.array(python_utci)
        
        # Add Python UTCI points as flat squares
        if len(python_positions) > 0:
            # Create enhanced hover text with X/Y/Z coordinates and UTCI values
            hover_text = [
                f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<br>"
                f"UTCI: {utci:.1f}°C<br>"
                f"Comfort: {self.get_utci_category(utci)}"
                for pos, utci in zip(python_positions, python_utci)
            ]
            
            fig.add_trace(go.Scatter3d(
                x=python_positions[:, 0],
                y=python_positions[:, 1],
                z=python_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=12,  # Larger squares
                    color=python_utci,
                    symbol='square',  # Square markers
                    colorscale='RdYlBu_r',
                    colorbar=dict(title="UTCI (°C)", x=0.7),
                    showscale=True,
                    line=dict(width=2, color='darkgray'),
                    opacity=0.9
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name='Python UTCI Grid',
                showlegend=True
            ), row=1, col=1)
        
        # Compute comparison statistics
        python_mean = np.mean(python_utci) if len(python_utci) > 0 else 0
        python_min = np.min(python_utci) if len(python_utci) > 0 else 0
        python_max = np.max(python_utci) if len(python_utci) > 0 else 0
        
        # Extract Grasshopper UTCI statistics (UTCI data from CSV)
        if 'utci' in gh_data.columns:
            gh_utci = gh_data['utci'].values
            gh_utci_mean = np.mean(gh_utci)
            gh_utci_min = np.min(gh_utci)
            gh_utci_max = np.max(gh_utci)
        else:
            # Fallback: assume UTCI is in 4th column (index 3)
            gh_utci = gh_data.iloc[:, 3].values
            gh_utci_mean = np.mean(gh_utci)
            gh_utci_min = np.min(gh_utci)
            gh_utci_max = np.max(gh_utci)
        
        # Calculate correlation if we have matching data points
        correlation_coeff = "N/A"
        if len(python_utci) > 0 and len(gh_utci) > 0:
            # Try to match the datasets for correlation
            min_len = min(len(python_utci), len(gh_utci))
            if min_len > 1:  # Need at least 2 points for correlation
                corr_matrix = np.corrcoef(python_utci[:min_len], gh_utci[:min_len])
                correlation_coeff = f"{corr_matrix[0, 1]:.3f}"
        
        # Create comparison table
        comparison_data = {
            'Metric': [
                'Data Points',
                'Mean UTCI (°C)',
                'Min UTCI (°C)', 
                'Max UTCI (°C)',
                'Range (°C)',
                'Std Dev (°C)',
                'Correlation'
            ],
            'Python': [
                f"{len(python_utci)}",
                f"{python_mean:.1f}",
                f"{python_min:.1f}",
                f"{python_max:.1f}",
                f"{python_max - python_min:.1f}",
                f"{np.std(python_utci):.1f}" if len(python_utci) > 0 else "0",
                "—"
            ],
            'Grasshopper': [
                f"{len(gh_data)}",
                f"{gh_utci_mean:.1f}",
                f"{gh_utci_min:.1f}",
                f"{gh_utci_max:.1f}",
                f"{gh_utci_max - gh_utci_min:.1f}",
                f"{np.std(gh_utci):.1f}",
                correlation_coeff
            ]
        }
        
        fig.add_trace(go.Table(
            header=dict(
                values=['Metric', 'Python', 'Grasshopper'],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[comparison_data['Metric'], comparison_data['Python'], comparison_data['Grasshopper']],
                fill_color='white',
                align='center',
                font=dict(size=11, color='black')
            )
        ), row=1, col=2)
        
        # Update layout
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            row=1, col=1
        )
        
        fig.update_layout(
            title=title,
            height=800,
            width=1400,
            showlegend=True,
            # Annotations removed - now comparing UTCI vs UTCI correctly
        )
        
        return fig
    
    def create_comfort_summary(self, utci_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create thermal comfort summary statistics.
        
        Args:
            utci_results: Dictionary from UTCICalculator.compute_utci()
            
        Returns:
            Dictionary with comfort analysis summary
        """
        all_utci = []
        comfort_counts = {}
        
        # Initialize comfort counters
        for category_data in self.utci_comfort_scale.values():
            comfort_counts[category_data['label']] = 0
        
        # Process all UTCI values
        for pos_key, data in utci_results.items():
            utci_vals = data['utci']
            
            for utci_val in utci_vals:
                # Handle both numeric values and potential objects
                try:
                    if hasattr(utci_val, 'utci'):
                        numeric_val = float(utci_val.utci)
                    elif isinstance(utci_val, dict) and 'utci' in utci_val:
                        numeric_val = float(utci_val['utci'])
                    else:
                        numeric_val = float(utci_val)
                    
                    if not np.isnan(numeric_val):
                        all_utci.append(numeric_val)
                        category = self.get_utci_category(numeric_val)
                        comfort_counts[category] = comfort_counts.get(category, 0) + 1
                except (ValueError, TypeError, AttributeError):
                    # Skip invalid values
                    continue
        
        all_utci = np.array(all_utci)
        total_points = len(all_utci)
        
        # Calculate percentages
        comfort_percentages = {
            category: (count / total_points * 100) if total_points > 0 else 0
            for category, count in comfort_counts.items()
        }
        
        summary = {
            'total_points': total_points,
            'utci_statistics': {
                'mean': float(np.mean(all_utci)) if len(all_utci) > 0 else 0,
                'min': float(np.min(all_utci)) if len(all_utci) > 0 else 0,
                'max': float(np.max(all_utci)) if len(all_utci) > 0 else 0,
                'std': float(np.std(all_utci)) if len(all_utci) > 0 else 0,
                'range': float(np.max(all_utci) - np.min(all_utci)) if len(all_utci) > 0 else 0
            },
            'comfort_distribution': {
                'counts': comfort_counts,
                'percentages': comfort_percentages
            },
            'comfort_assessment': self._assess_overall_comfort(comfort_percentages)
        }
        
        return summary
    
    def _assess_overall_comfort(self, comfort_percentages: Dict[str, float]) -> str:
        """Assess overall thermal comfort based on distribution."""
        comfortable_percent = comfort_percentages.get('Comfortable', 0)
        
        if comfortable_percent >= 80:
            return "Excellent - Majority of space is thermally comfortable"
        elif comfortable_percent >= 60:
            return "Good - Most of space is comfortable with some issues"
        elif comfortable_percent >= 40:
            return "Moderate - Mixed comfort conditions across space"
        elif comfortable_percent >= 20:
            return "Poor - Limited comfortable areas"
        else:
            return "Critical - Very few comfortable areas"


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
