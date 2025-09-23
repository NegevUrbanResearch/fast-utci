"""
Enhanced 3D Viewer for fast-utci

This module provides enhanced 3D visualization capabilities with layer support,
better curve/road visualization, and improved model display.
"""

import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from ladybug_utci_colors import LadybugUTCIColors, create_ladybug_utci_colorscale
from enhanced_model_reader import EnhancedModel, ModelLayer


class EnhancedUTCIViewer:
    """
    Enhanced 3D viewer for UTCI heatmap visualization with layer support.
    
    Provides improved visualization with:
    - Layer-based model display with different colors
    - Better curve/road visualization
    - Static UTCI color scale
    - Enhanced model context
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ladybug UTCI color scale
        self.utci_colors = LadybugUTCIColors()
        self.static_colorscale = create_ladybug_utci_colorscale()
        self.colorscale_bounds = self.utci_colors.get_colorscale_bounds()
    
    def visualize_enhanced_utci_heatmap(self, 
                                      enhanced_model: EnhancedModel,
                                      utci_results: Dict[str, Any],
                                      title: str = "Enhanced UTCI Thermal Comfort Heatmap",
                                      show_model_layers: bool = True,
                                      show_utci_points: bool = True,
                                      show_comfort_legend: bool = True,
                                      point_size: int = 8,
                                      analysis_type: str = "single_hour",
                                      validation_csv: str = None) -> go.Figure:
        """
        Create enhanced 3D visualization of UTCI heatmap with layer support.
        
        Args:
            enhanced_model: EnhancedModel with layer information
            utci_results: Dictionary from UTCICalculator.compute_utci()
            title: Plot title
            show_model_layers: Whether to show model layers
            show_utci_points: Whether to show UTCI data points
            show_comfort_legend: Whether to show thermal comfort legend
            point_size: Size of UTCI data points
            analysis_type: "single_hour" for dynamic colorscale, "full_day" for static
            validation_csv: Path to validation CSV for comparison table
            
        Returns:
            Plotly figure with enhanced 3D UTCI heatmap
        """
        fig = go.Figure()
        
        # Determine colorscale and bounds based on analysis type
        if analysis_type == "single_hour":
            # Dynamic colorscale for single hour - use actual data range
            # Extract UTCI values from all positions
            all_utci_values = []
            for pos_key, pos_data in utci_results.items():
                if 'utci' in pos_data:
                    all_utci_values.extend(pos_data['utci'])
            utci_values = np.array(all_utci_values)
            utci_min, utci_max = float(np.min(utci_values)), float(np.max(utci_values))
            colorscale = self.utci_colors.create_dynamic_colorscale(utci_min, utci_max)
            colorscale_bounds = (utci_min, utci_max)  # Use actual data range for maximum contrast
            print(f"🎨 Using dynamic colorscale for single hour: full spectrum mapped to data range {utci_min:.1f}°C to {utci_max:.1f}°C")
        else:
            # Static colorscale for full day analysis
            colorscale = self.static_colorscale
            colorscale_bounds = self.colorscale_bounds
            print(f"🎨 Using static colorscale for full day: {colorscale_bounds[0]:.1f}°C to {colorscale_bounds[1]:.1f}°C")
        
        # Add model layers if requested
        if show_model_layers:
            self._add_model_layers(fig, enhanced_model)
        
        # Add UTCI data points if requested
        if show_utci_points:
            self._add_utci_points(fig, utci_results, point_size, enhanced_model, colorscale, colorscale_bounds)
        
        # Add comfort legend if requested
        if show_comfort_legend:
            self._add_comfort_legend(fig, utci_results)
        
        # Add comparison table if validation CSV is provided
        if validation_csv and Path(validation_csv).exists():
            self._add_comparison_table(fig, utci_results, validation_csv, analysis_type)
        
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
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                zaxis=dict(showgrid=False),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)
                )
            ),
            width=1400,
            height=900,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )
        )
        
        return fig
    
    def _add_model_layers(self, fig: go.Figure, enhanced_model: EnhancedModel):
        """Add consolidated model layers to the figure with different colors."""
        # Group layers by material type to reduce the number of traces
        layers_by_type = {}
        
        for layer in enhanced_model.layers:
            material_type = layer.material_type
            if material_type not in layers_by_type:
                layers_by_type[material_type] = []
            layers_by_type[material_type].append(layer)
        
        # Add one trace per material type (consolidated)
        for material_type, layers in layers_by_type.items():
            if not layers:
                continue
                
            # Get material info from first layer of this type
            material_info = layers[0].material_info
            
            # Combine all meshes of this type
            combined_vertices = []
            combined_faces = []
            face_offset = 0
            total_vertices = 0
            total_faces = 0
            
            for layer in layers:
                mesh = layer.mesh
                combined_vertices.append(mesh.vertices)
                
                # Adjust face indices for combined mesh
                adjusted_faces = mesh.faces + face_offset
                combined_faces.append(adjusted_faces)
                
                face_offset += len(mesh.vertices)
                total_vertices += len(mesh.vertices)
                total_faces += len(mesh.faces)
            
            if combined_vertices:
                # Combine all vertices and faces
                all_vertices = np.vstack(combined_vertices)
                all_faces = np.vstack(combined_faces)
                
                # Add consolidated mesh trace
                # Add main mesh
                fig.add_trace(go.Mesh3d(
                    x=all_vertices[:, 0],
                    y=all_vertices[:, 1],
                    z=all_vertices[:, 2],
                    i=all_faces[:, 0],
                    j=all_faces[:, 1],
                    k=all_faces[:, 2],
                    opacity=material_info['opacity'],
                    color=material_info['color'],
                    name=material_info['name'],
                    showlegend=True,
                    lighting=dict(
                        ambient=0.4,
                        diffuse=1.0,
                        specular=0.2,
                        roughness=0.1,
                        fresnel=0.2
                    ),
                    lightposition=dict(x=100, y=200, z=300),
                     hovertemplate=f"<b>{material_info['name']}</b><br>" +
                                  f"Objects: {len(layers)}<br>" +
                                  f"Vertices: {total_vertices:,}<br>" +
                                  f"Faces: {total_faces:,}<br>" +
                                  f"Material: {material_type}<br>" +
                                  f"Height Range: {self._get_height_range(all_vertices):.1f}m<extra></extra>",
                     hoverinfo="skip"  # Skip hover for model layers to prioritize UTCI tooltips
                ))
                
                # Add dark edges for white buildings to ensure visibility
                if material_type == 'building' and material_info['color'] == 'white':
                    # Use a darker color for the building edges instead of line property
                    fig.add_trace(go.Mesh3d(
                        x=all_vertices[:, 0],
                        y=all_vertices[:, 1],
                        z=all_vertices[:, 2],
                        i=all_faces[:, 0],
                        j=all_faces[:, 1],
                        k=all_faces[:, 2],
                        opacity=0.1,  # Very low opacity for subtle edge effect
                        color='darkgray',  # Dark gray for edges
                        name=f"{material_info['name']} Edges",
                        showlegend=False,
                        lighting=dict(
                            ambient=0.4,
                            diffuse=1.0,
                            specular=0.2,
                            roughness=0.1,
                            fresnel=0.2
                        ),
                        lightposition=dict(x=100, y=200, z=300),
                        hovertemplate="",
                        hoverinfo="skip",
                        flatshading=True
                    ))
    
    def _get_height_range(self, vertices: np.ndarray) -> float:
        """Calculate the height range (max - min Z) of vertices."""
        if len(vertices) == 0:
            return 0.0
        z_coords = vertices[:, 2]
        return float(np.max(z_coords) - np.min(z_coords))
    
    def _add_utci_points(self, fig: go.Figure, utci_results: Dict[str, Any], point_size: int, enhanced_model=None, colorscale=None, colorscale_bounds=None):
        """Add UTCI data points to the figure with both square grid and interpolation layers."""
        # Extract UTCI data points
        positions = []
        utci_values = []
        mrt_values = []
        categories = []
        
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
                        categories.append(self.utci_colors.get_utci_abbrev(utci_val))
        
        if len(positions) == 0:
            print("Warning: No valid UTCI data points to display")
            return
        
        positions = np.array(positions)
        utci_values = np.array(utci_values)
        mrt_values = np.array(mrt_values)
        
        # Create hover text for square grid points
        hover_text = [
            f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<br>"
            f"UTCI: {utci:.1f}°C<br>"
            f"MRT: {mrt:.1f}°C<br>"
            f"Comfort: {cat}"
            for pos, utci, mrt, cat in zip(positions, utci_values, mrt_values, categories)
        ]
        
        # Add UTCI as true 2D plane using surface plot (always faces up)
        base_z_level = -10.0  # Lower Z level to ensure UTCI is below model geometry
        
        # Use base layer bounds instead of UTCI data bounds for proper alignment
        if enhanced_model:
            # Get bounds from base layer
            base_bounds = enhanced_model.get_bounds_for_layer_type('base')
            if base_bounds is not None:
                x_min, y_min = base_bounds[0][:2]  # Use X,Y from base bounds
                x_max, y_max = base_bounds[1][:2]
                print(f"🗺️ Using base layer bounds for UTCI: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
            else:
                # Fallback to UTCI data bounds
                x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
                print(f"⚠️ No base layer found, using UTCI data bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
        else:
            # Fallback to UTCI data bounds
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            print(f"⚠️ No enhanced model provided, using UTCI data bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
        
        # Use passed colorscale or default to static
        if colorscale is None:
            colorscale = self.static_colorscale
        if colorscale_bounds is None:
            colorscale_bounds = self.colorscale_bounds
        
        # Ensure colorscale bounds match the actual data range
        valid_utci = utci_values[~np.isnan(utci_values)]
        if len(valid_utci) > 0:
            actual_min = float(np.min(valid_utci))
            actual_max = float(np.max(valid_utci))
            colorscale_bounds = (actual_min, actual_max)
            print(f"🎨 Adjusted colorscale bounds to match data: {actual_min:.1f}°C to {actual_max:.1f}°C")
        else:
            print("⚠️ No valid UTCI data found for colorscale bounds")
        
        # Filter square grid points to match interpolation bounds
        # Only show points within the same bounds as the interpolation layer
        valid_indices = []
        for i, pos in enumerate(positions):
            if (x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max):
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            print("Warning: No UTCI points within model bounds for square grid")
            return
        
        # Filter data to only include points within bounds
        filtered_positions = positions[valid_indices]
        filtered_utci_values = utci_values[valid_indices]
        filtered_hover_text = [hover_text[i] for i in valid_indices]
        
        # Update colorscale bounds based on filtered data
        valid_filtered_utci = filtered_utci_values[~np.isnan(filtered_utci_values)]
        if len(valid_filtered_utci) > 0:
            filtered_min = float(np.min(valid_filtered_utci))
            filtered_max = float(np.max(valid_filtered_utci))
            colorscale_bounds = (filtered_min, filtered_max)
            print(f"🎨 Final colorscale bounds from filtered data: {filtered_min:.1f}°C to {filtered_max:.1f}°C")
        
        print(f"🔲 Square grid: {len(filtered_positions)} points within model bounds (filtered from {len(positions)} total)")
        
        # LAYER 1: Square Grid (discrete points) - SHOWN BY DEFAULT
        # Use Scatter3d with square markers - they face camera but are discrete
        fig.add_trace(go.Scatter3d(
            x=filtered_positions[:, 0],
            y=filtered_positions[:, 1],
            z=np.full(len(filtered_positions), base_z_level),  # All points at same Z level
            mode='markers',
            marker=dict(
                size=point_size,
                color=filtered_utci_values,
                colorscale=colorscale,
                cmin=colorscale_bounds[0],
                cmax=colorscale_bounds[1],
                symbol='square',  # Use square markers
                line=dict(width=0),
                showscale=True,  # Show colorbar for this layer
                colorbar=dict(
                    title=dict(text="UTCI (°C)", side="right"),
                    tickmode="array",
                    tickvals=[colorscale_bounds[0] + i * (colorscale_bounds[1] - colorscale_bounds[0]) / 10 for i in range(11)],
                    ticktext=[f"{colorscale_bounds[0] + i * (colorscale_bounds[1] - colorscale_bounds[0]) / 10:.1f}" for i in range(11)]
                )
            ),
            hovertext=filtered_hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            name='UTCI Square Grid',
            showlegend=True,
            visible=True  # Shown by default
        ))
        
        print(f"🔲 Square grid: {len(filtered_positions)} discrete points with square markers")
        
        # LAYER 2: Interpolation Surface - HIDDEN BY DEFAULT
        # Create interpolation grid with higher resolution for better coverage
        # Adaptive resolution for smoother interpolation while keeping perf reasonable
        grid_resolution = int(min(300, max(50, np.sqrt(len(filtered_positions)) * 2)))
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate UTCI values to grid using linear interpolation for smooth surface
        # Use the same filtered data as the square grid for consistency
        from scipy.interpolate import griddata
        Z_utci = griddata(
            (filtered_positions[:, 0], filtered_positions[:, 1]), 
            filtered_utci_values, 
            (X, Y), 
            method='linear',  # Use linear interpolation for smooth surface
            fill_value=np.nan
        )
        
        # Fill any remaining NaN values at edges with nearest valid values
        if np.any(np.isnan(Z_utci)):
            valid_mask = ~np.isnan(Z_utci)
            if np.any(valid_mask):
                # Use nearest neighbor interpolation to fill edge gaps
                Z_utci_filled = griddata(
                    (X[valid_mask], Y[valid_mask]), 
                    Z_utci[valid_mask], 
                    (X, Y), 
                    method='nearest'
                )
                # Only replace NaN values, keep original interpolated values
                Z_utci = np.where(np.isnan(Z_utci), Z_utci_filled, Z_utci)
        
        # Create flat Z grid (all at same level)
        Z_flat = np.full_like(X, base_z_level)
        
        # Create custom hover text with UTCI values for interpolation layer
        hover_text_interp = np.empty_like(Z_utci, dtype=object)
        for i in range(Z_utci.shape[0]):
            for j in range(Z_utci.shape[1]):
                utci_val = Z_utci[i, j]
                if not np.isnan(utci_val):
                    category = self.utci_colors.get_utci_label(utci_val)
                    hover_text_interp[i, j] = f"<b>UTCI: {utci_val:.1f}°C</b><br>Category: {category}"
                else:
                    hover_text_interp[i, j] = "No data"
        
        # Add interpolation surface plot
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_flat,
            colorscale=colorscale,
            surfacecolor=Z_utci,
            cmin=colorscale_bounds[0],
            cmax=colorscale_bounds[1],
            opacity=0.7,  # Slightly higher for readability
            hovertext=hover_text_interp,
            hovertemplate='%{hovertext}<extra></extra>',
                colorbar=dict(
                    title=dict(text="UTCI (°C)", side="right"),
                    tickmode="array",
                    tickvals=[colorscale_bounds[0] + i * (colorscale_bounds[1] - colorscale_bounds[0]) / 10 for i in range(11)],
                    ticktext=[f"{colorscale_bounds[0] + i * (colorscale_bounds[1] - colorscale_bounds[0]) / 10:.1f}" for i in range(11)]
                ),
            name='UTCI Interpolation',
            showlegend=True,
            showscale=True,
            visible='legendonly'  # Hidden by default, can be toggled via legend
        ))
    
    def _add_comfort_legend(self, fig: go.Figure, utci_results: Dict[str, Any]):
        """Add comfort legend to the figure without discrete points."""
        # Note: This method is kept for compatibility but no longer adds discrete points
        # The UTCI colorbar in the surface plot provides the thermal comfort scale
        # Discrete points in the legend have been removed as requested
        pass
    
    def _add_comparison_table(self, fig: go.Figure, utci_results: Dict[str, Any], validation_csv: str, analysis_type: str):
        """Add comparison table with Grasshopper validation data."""
        try:
            # Load Grasshopper validation data
            gh_data = pd.read_csv(validation_csv)
            
            # Extract Python UTCI data
            python_utci = []
            for pos_key, data in utci_results.items():
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
                            python_utci.append(utci_val)
            
            python_utci = np.array(python_utci)
            
            # Compute comparison statistics
            python_mean = np.mean(python_utci) if len(python_utci) > 0 else 0
            python_min = np.min(python_utci) if len(python_utci) > 0 else 0
            python_max = np.max(python_utci) if len(python_utci) > 0 else 0
            
            # Extract Grasshopper UTCI statistics
            if 'utci' in gh_data.columns:
                gh_utci = gh_data['utci'].values
            else:
                # Fallback: assume UTCI is in 4th column (index 3)
                gh_utci = gh_data.iloc[:, 3].values
            
            gh_utci_mean = np.mean(gh_utci)
            gh_utci_min = np.min(gh_utci)
            gh_utci_max = np.max(gh_utci)
            
            # Calculate correlation if we have matching data points
            correlation_coeff = "N/A"
            if len(python_utci) > 0 and len(gh_utci) > 0:
                min_len = min(len(python_utci), len(gh_utci))
                if min_len > 1:
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
            
            # Add comparison table
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
                ),
                domain=dict(x=[0.0, 0.4], y=[0.0, 0.4])  # Position in bottom-left where there's plenty of space
            ))
            
            print(f"📊 Added comparison table: Python ({len(python_utci)} points) vs Grasshopper ({len(gh_data)} points)")
            print(f"   Correlation: {correlation_coeff}")
            
        except Exception as e:
            print(f"⚠️ Could not add comparison table: {e}")
    
    def create_animated_enhanced_visualization(self, 
                                             enhanced_model: EnhancedModel,
                                             utci_results: Dict[str, Any], 
                                             title: str = "Enhanced 24-Hour UTCI Analysis",
                                             analysis_type: str = "full_day") -> go.Figure:
        """
        Create animated UTCI visualization with enhanced model layers.
        
        Args:
            enhanced_model: EnhancedModel with layer information
            utci_results: UTCI results dictionary
            title: Plot title
            
        Returns:
            Plotly figure with animation
        """
        import plotly.graph_objects as go
        
        # Create figure
        fig = go.Figure()
        
        # Determine colorscale and bounds based on analysis type
        if analysis_type == "single_hour":
            # Dynamic colorscale for single hour - use actual data range
            # Extract UTCI values from all positions
            all_utci_values = []
            for pos_key, pos_data in utci_results.items():
                if 'utci' in pos_data:
                    all_utci_values.extend(pos_data['utci'])
            utci_values = np.array(all_utci_values)
            utci_min, utci_max = float(np.min(utci_values)), float(np.max(utci_values))
            colorscale = self.utci_colors.create_dynamic_colorscale(utci_min, utci_max)
            colorscale_bounds = (utci_min, utci_max)  # Use actual data range for maximum contrast
            print(f"🎨 Using dynamic colorscale for single hour animation: full spectrum mapped to data range {utci_min:.1f}°C to {utci_max:.1f}°C")
        else:
            # Static colorscale for full day analysis
            colorscale = self.static_colorscale
            colorscale_bounds = self.colorscale_bounds
            print(f"🎨 Using static colorscale for full day animation: {colorscale_bounds[0]:.1f}°C to {colorscale_bounds[1]:.1f}°C")
        
        # Add model layers (static background)
        self._add_model_layers(fig, enhanced_model)
        
        # Extract UTCI data by hour
        utci_data_by_hour = {}
        hours_seen = set()
        
        for pos_key, data in utci_results.items():
            position = np.asarray(data['position'])
            utci_vals = data['utci']
            datetimes = data.get('datetime', None)
            
            if isinstance(utci_vals, (list, np.ndarray)) and len(utci_vals) > 0:
                for idx, utci_val in enumerate(utci_vals):
                    # Determine hour label
                    hour = None
                    try:
                        if datetimes is not None and idx < len(datetimes) and datetimes[idx] is not None:
                            hour = int(pd.to_datetime(datetimes[idx]).hour)
                        else:
                            hour = idx
                    except Exception:
                        hour = idx
                    
                    hours_seen.add(hour)
                    if hour not in utci_data_by_hour:
                        utci_data_by_hour[hour] = {'positions': [], 'utci_values': []}
                    
                    # Extract numeric UTCI value
                    try:
                        if hasattr(utci_val, 'utci'):
                            numeric_utci = float(utci_val.utci)
                        elif isinstance(utci_val, dict) and 'utci' in utci_val:
                            numeric_utci = float(utci_val['utci'])
                        else:
                            numeric_utci = float(utci_val)
                        if not np.isnan(numeric_utci):
                            utci_data_by_hour[hour]['positions'].append(position)
                            utci_data_by_hour[hour]['utci_values'].append(numeric_utci)
                    except (ValueError, TypeError, AttributeError):
                        continue
        
        # Create frames for animation
        frames = []
        available_hours = sorted(list(hours_seen)) if len(hours_seen) > 0 else list(range(24))
        base_z_level = -5.0  # Fixed Z level for 2D plane
        
        for hour in available_hours:
            if hour in utci_data_by_hour and len(utci_data_by_hour[hour]['positions']) > 0:
                positions = np.array(utci_data_by_hour[hour]['positions'])
                utci_values = np.array(utci_data_by_hour[hour]['utci_values'])
                
                # Create hover text
                hover_text = [
                    f"Hour: {hour:02d}:00<br>"
                    f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<br>"
                    f"UTCI: {utci:.1f}°C"
                    for pos, utci in zip(positions, utci_values)
                ]
                
                # Use base layer bounds for proper alignment (same as static visualization)
                base_bounds = enhanced_model.get_bounds_for_layer_type('base')
                if base_bounds is not None:
                    x_min, y_min = base_bounds[0][:2]  # Use X,Y from base bounds
                    x_max, y_max = base_bounds[1][:2]
                else:
                    # Fallback to UTCI data bounds
                    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
                    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
                
                grid_resolution = 15  # Slightly lower for animation performance
                x_grid = np.linspace(x_min, x_max, grid_resolution)
                y_grid = np.linspace(y_min, y_max, grid_resolution)
                X, Y = np.meshgrid(x_grid, y_grid)
                
                # Interpolate UTCI values to grid using nearest neighbor for discrete boundaries
                from scipy.interpolate import griddata
                Z_utci = griddata(
                    (positions[:, 0], positions[:, 1]), 
                    utci_values, 
                    (X, Y), 
                    method='nearest',  # Use nearest neighbor for discrete boundaries
                    fill_value=np.nan
                )
                
                Z_flat = np.full_like(X, base_z_level)
                
                frame_data = [
                    go.Surface(
                        x=X, y=Y, z=Z_flat,
                        colorscale=colorscale,
                        surfacecolor=Z_utci,
                        cmin=colorscale_bounds[0],
                        cmax=colorscale_bounds[1],
                        opacity=0.8,
                        showscale=False,
                        name=f'Hour {hour:02d}:00',
                        showlegend=False
                    )
                ]
                
                # Count model layers to determine trace index
                num_model_traces = len(enhanced_model.layers)
                frame = go.Frame(
                    data=frame_data,
                    traces=[num_model_traces],  # Update the UTCI trace (after model layers)
                    name=f"frame_{hour}"
                )
                frames.append(frame)
        
        fig.frames = frames
        
        # Add initial scatter for the first available hour
        initial_hour = available_hours[0] if len(available_hours) > 0 else 0
        if initial_hour in utci_data_by_hour and len(utci_data_by_hour[initial_hour]['positions']) > 0:
            init_positions = np.array(utci_data_by_hour[initial_hour]['positions'])
            init_utci = np.array(utci_data_by_hour[initial_hour]['utci_values'])
            init_hover = [
                f"Hour: {initial_hour:02d}:00<br>"
                f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<br>"
                f"UTCI: {val:.1f}°C" for pos, val in zip(init_positions, init_utci)
            ]
            # Create 2D plane using surface plot for initial frame
            x_min, x_max = init_positions[:, 0].min(), init_positions[:, 0].max()
            y_min, y_max = init_positions[:, 1].min(), init_positions[:, 1].max()
            
            grid_resolution = 50  # Increased for better edge coverage
            x_grid = np.linspace(x_min, x_max, grid_resolution)
            y_grid = np.linspace(y_min, y_max, grid_resolution)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            from scipy.interpolate import griddata
            Z_utci = griddata(
                (init_positions[:, 0], init_positions[:, 1]), 
                init_utci, 
                (X, Y), 
                method='linear',
                fill_value=np.nan
            )
            
            # Fill any remaining NaN values at edges with nearest valid values
            if np.any(np.isnan(Z_utci)):
                valid_mask = ~np.isnan(Z_utci)
                if np.any(valid_mask):
                    Z_utci_filled = griddata(
                        (X[valid_mask], Y[valid_mask]), 
                        Z_utci[valid_mask], 
                        (X, Y), 
                        method='nearest'
                    )
                    Z_utci = np.where(np.isnan(Z_utci), Z_utci_filled, Z_utci)
            
            Z_flat = np.full_like(X, base_z_level)
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z_flat,
                colorscale=colorscale,
                surfacecolor=Z_utci,
                cmin=colorscale_bounds[0],
                cmax=colorscale_bounds[1],
                opacity=0.8,
                colorbar=dict(
                    title=dict(text="UTCI (°C)"),
                    tickmode="array",
                    tickvals=[colorscale_bounds[0] + i * (colorscale_bounds[1] - colorscale_bounds[0]) / 10 for i in range(11)],
                    ticktext=[f"{colorscale_bounds[0] + i * (colorscale_bounds[1] - colorscale_bounds[0]) / 10:.1f}" for i in range(11)]
                ),
                showscale=True,
                name='UTCI Base Layer (2D)',
                showlegend=True
            ))
        
        # Add animation controls
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1400,
            height=900,
            updatemenus=[
                {
                    'type': 'buttons',
                    'showactive': True,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'frame': {'duration': 800, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 200, 'easing': 'linear'}
                                }
                            ]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [
                                [[None]],
                                {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ]
                        }
                    ],
                    'x': 0.05,
                    'y': 0.05
                }
            ],
            sliders=[
                {
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'prefix': 'Hour: ',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': [
                        {
                            'args': [
                                [f"frame_{hour}"],
                                {
                                    'frame': {'duration': 0, 'redraw': True},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ],
                            'label': f'{hour:02d}:00',
                            'method': 'animate'
                        }
                        for hour in available_hours
                    ]
                }
            ]
        )
        
        # Set camera position
        fig.update_layout(
            scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        )
        
        return fig


# Example usage
if __name__ == "__main__":
    # Test the enhanced viewer
    print("Enhanced UTCI Viewer module loaded successfully!")
    print("Use EnhancedUTCIViewer class for improved visualization with layer support.")
