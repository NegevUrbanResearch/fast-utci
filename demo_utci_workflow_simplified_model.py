"""
Complete UTCI Workflow Demonstration for fast-utci.

This script demonstrates the full pipeline:
1. Load 3D model and weather data using reader.py
2. Compute MRT using MRT calculator with parallel processing
3. Compute UTCI using UTCI calculator
4. Visualize results with 3D heatmap viewer
5. Compare with Grasshopper validation data
"""

from pathlib import Path
import numpy as np
from typing import Tuple, List, Optional
import pandas as pd
import psutil
import gc

def get_user_analysis_choice():
    """Get user choice for analysis type."""
    print("\n" + "="*50)
    print("ANALYSIS MODE SELECTION")
    print("="*50)
    print("1. Single Hour Analysis")
    print("2. Full Day Analysis (24 hours)")
    print("="*50)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "single_hour"
        elif choice == "2":
            return "full_day"
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")


def get_single_hour_input():
    """Get hour input for single hour analysis."""
    print("\n" + "="*40)
    print("SINGLE HOUR ANALYSIS")
    print("="*40)
    print("Enter the hour to analyze (0-23):")
    print("  - 0 = Midnight (00:00-01:00)")
    print("  - 12 = Noon (12:00-13:00)")
    print("  - 13 = 1 PM (13:00-14:00) - Default validation hour")
    print("  - 23 = 11 PM (23:00-24:00)")
    
    while True:
        try:
            hour_input = input("Hour (0-23, or press Enter for default 13): ").strip()
            if hour_input == "":
                return 13  # Default validation hour
            hour = int(hour_input)
            if 0 <= hour <= 23:
                return hour
            else:
                print("❌ Hour must be between 0 and 23.")
        except ValueError:
            print("❌ Please enter a valid number between 0 and 23.")


def monitor_memory_usage():
    """Monitor and display current memory usage."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"💾 Memory usage: {memory_mb:.1f} MB")
        return memory_mb
    except Exception:
        return 0


def cleanup_memory():
    """Force garbage collection to free up memory."""
    gc.collect()


def validate_analysis_mode(analysis_mode: str, target_hour: Optional[int] = None) -> bool:
    """
    Validate that the system can handle the requested analysis mode.
    
    Args:
        analysis_mode: "single_hour" or "full_day"
        target_hour: Hour for single hour analysis
        
    Returns:
        True if analysis mode is valid
    """
    if analysis_mode not in ["single_hour", "full_day"]:
        print(f"❌ Invalid analysis mode: {analysis_mode}")
        return False
    
    if analysis_mode == "single_hour" and target_hour is not None:
        if not (0 <= target_hour <= 23):
            print(f"❌ Invalid target hour: {target_hour}. Must be 0-23.")
            return False
    
    # Check available memory for full day analysis
    if analysis_mode == "full_day":
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2.0:  # Less than 2GB available
                print(f"⚠️  Warning: Low available memory ({available_memory_gb:.1f} GB)")
                print("   Full day analysis may be slow or fail. Consider closing other applications.")
                
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("❌ Analysis cancelled by user.")
                    return False
        except Exception:
            print("⚠️  Could not check available memory.")
    
    return True


def create_analysis_period_and_hours(analysis_mode: str, target_hour: Optional[int] = None) -> Tuple[any, List[int]]:
    """
    Create analysis period and target hours based on analysis mode.
    
    Args:
        analysis_mode: "single_hour" or "full_day"
        target_hour: Hour for single hour analysis (0-23)
        
    Returns:
        Tuple of (analysis_period, target_hours)
    """
    from MRT.period import create_analysis_period
    
    if analysis_mode == "single_hour":
        # Single hour analysis
        if target_hour is None:
            target_hour = 13  # Default validation hour
        
        analysis_period = create_analysis_period(
            start_month=8, start_day=15,
            end_month=8, end_day=15,
            start_hour=target_hour, end_hour=target_hour
        )
        target_hours = [target_hour]
        
        print(f"📅 Analysis period: August 15th, hour {target_hour:02d}:00")
        
    else:  # full_day
        # Full day analysis (24 hours)
        analysis_period = create_analysis_period(
            start_month=8, start_day=15,
            end_month=8, end_day=15,
            start_hour=0, end_hour=23
        )
        target_hours = list(range(24))  # [0, 1, 2, ..., 23]
        
        print(f"📅 Analysis period: August 15th, full day (00:00-24:00)")
        print(f"⏰ Target hours: {len(target_hours)} hours (0-23)")
    
    return analysis_period, target_hours


def create_visualization(analysis_mode: str, model, utci_results: dict, validation_csv: str, grid_size: float) -> str:
    """
    Create appropriate visualization based on analysis mode.
    
    Args:
        analysis_mode: "single_hour" or "full_day"
        model: 3D model mesh
        utci_results: UTCI calculation results
        validation_csv: Path to validation CSV
        grid_size: Grid spacing for filename
        
    Returns:
        Filename of created visualization
    """
    from viewer import UTCIHeatmapViewer
    
    viewer = UTCIHeatmapViewer()
    
    if analysis_mode == "single_hour":
        # Single hour visualization (existing functionality)
        print("📊 Creating single hour comparison with Grasshopper validation data...")
        
        comparison_fig = viewer.compare_with_grasshopper(
            model_mesh=model,
            utci_results=utci_results,
            grasshopper_csv=validation_csv,
            title="UTCI Results: Python (70% Simplified Model) vs Grasshopper Validation - Single Hour"
        )
        
        # Save the comparison as HTML file
        comparison_filename = f"utci_comparison_grid_{grid_size}m_simplified_70pct_single_hour.html"
        
    else:  # full_day
        # Full day animated visualization
        print("📊 Creating animated 24-hour UTCI visualization...")
        
        comparison_fig = create_animated_utci_visualization(
            model_mesh=model,
            utci_results=utci_results,
            title="UTCI Results: 24-Hour Analysis - Python (70% Simplified Model)"
        )
        
        # Save the animated visualization as HTML file
        comparison_filename = f"utci_comparison_grid_{grid_size}m_simplified_70pct_24hour_animated.html"
    
    comparison_fig.write_html(comparison_filename)
    print(f"💾 Visualization saved: {comparison_filename}")
    
    return comparison_filename


def create_animated_utci_visualization(model_mesh, utci_results: dict, title: str = "24-Hour UTCI Analysis"):
    """
    Create animated UTCI visualization with time slider for 24-hour results.
    
    Args:
        model_mesh: 3D model mesh
        utci_results: UTCI results dictionary
        title: Plot title
        
    Returns:
        Plotly figure with animation
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Create figure
    fig = go.Figure()
    
    # Add 3D model (static background)
    if model_mesh is not None:
        fig.add_trace(go.Mesh3d(
            x=model_mesh.vertices[:, 0],
            y=model_mesh.vertices[:, 1],
            z=model_mesh.vertices[:, 2],
            i=model_mesh.faces[:, 0],
            j=model_mesh.faces[:, 1],
            k=model_mesh.faces[:, 2],
            opacity=0.6,
            color='darkslategray',
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
        ))
    
    # Extract UTCI data by hour using datetime when available for robust mapping
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
                        # pandas Timestamp or datetime64 -> extract hour
                        hour = int(pd.to_datetime(datetimes[idx]).hour)
                    else:
                        hour = idx  # fallback to sequence index
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
            
            frame_data = [
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        symbol='square',
                        color=utci_values,
                        colorscale='RdYlBu_r',
                        showscale=False,
                        line=dict(width=2, color='darkgray'),
                        opacity=0.9
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name=f'Hour {hour:02d}:00',
                    showlegend=False
                )
            ]
            
            frame = go.Frame(
                data=frame_data,
                traces=[1],  # update the second trace (index 1) which is the UTCI scatter; keep mesh (index 0)
                name=f"frame_{hour}"
            )
            frames.append(frame)
    
    fig.frames = frames
    
    # Add initial scatter for the first available hour so colorbar shows
    initial_hour = available_hours[0] if len(available_hours) > 0 else 0
    if initial_hour in utci_data_by_hour and len(utci_data_by_hour[initial_hour]['positions']) > 0:
        init_positions = np.array(utci_data_by_hour[initial_hour]['positions'])
        init_utci = np.array(utci_data_by_hour[initial_hour]['utci_values'])
        init_hover = [
            f"Hour: {initial_hour:02d}:00<br>"
            f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<br>"
            f"UTCI: {val:.1f}°C" for pos, val in zip(init_positions, init_utci)
        ]
        fig.add_trace(go.Scatter3d(
            x=init_positions[:, 0],
            y=init_positions[:, 1],
            z=init_positions[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                symbol='square',
                color=init_utci,
                colorscale='RdYlBu_r',
                colorbar=dict(title="UTCI (°C)"),
                showscale=True,
                line=dict(width=2, color='darkgray'),
                opacity=0.9
            ),
            text=init_hover,
            hovertemplate='%{text}<extra></extra>',
            name=f'Hour {initial_hour:02d}:00',
            showlegend=False
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
        width=1200,
        height=800,
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
    
    # Set a sensible camera so model and grid are visible
    fig.update_layout(
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
    )

    # Hide legend (to avoid accidental toggling of the animated trace) but keep colorbar
    fig.update_layout(showlegend=False)
    
    return fig


def main():
    """Run the complete UTCI workflow demonstration."""
    
    print("=" * 60)
    print("FAST-UTCI COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # Get user choice for analysis type
    analysis_mode = get_user_analysis_choice()
    
    if analysis_mode == "single_hour":
        target_hour = get_single_hour_input()
        print(f"✅ Selected: Single hour analysis for hour {target_hour:02d}:00")
    else:
        print("✅ Selected: Full day analysis (24 hours)")
        target_hour = None
    
    # Validate analysis mode
    if not validate_analysis_mode(analysis_mode, target_hour):
        return 1
    
    # File paths
    model_file = "data/rec_model_no_curve.glb"
    epw_file = "data/ISR_Beer.Sheva.401900_MSI.epw" 
    validation_csv = "data/15th_Aug_MRT.csv"
    
    # Check if files exist
    for file_path, name in [(model_file, "3D model"), (epw_file, "EPW weather"), (validation_csv, "validation CSV")]:
        if not Path(file_path).exists():
            print(f"❌ {name} file not found: {file_path}")
            return 1
    
    print(f"✅ All required files found")
    print(f"  Model: {model_file}")
    print(f"  Weather: {epw_file}")
    print(f"  Validation: {validation_csv}")
    
    try:
        # Step 1: Load data using reader module
        print("\n" + "="*40)
        print("STEP 1: LOADING PROJECT DATA")
        print("="*40)
        
        from reader import read_project_data
        model, weather_df, epw_data = read_project_data(model_file, epw_file)
        
        print(f"📊 Original model loaded: {len(model.vertices):,} vertices, {len(model.faces):,} faces")
        
        # Simplify model to 70% for performance testing
        print("🔧 Simplifying model to 70% for performance comparison...")
        import trimesh
        target_faces = int(len(model.faces) * 0.7)
        model = model.simplify_quadric_decimation(face_count=target_faces)
        print(f"📊 Simplified model: {len(model.vertices):,} vertices, {len(model.faces):,} faces")
        original_faces = len(model.faces) / 0.7  # Calculate original face count
        speedup = original_faces / len(model.faces)
        print(f"⚡ Expected ray casting speedup: ~{speedup:.1f}x faster")
        print(f"🌤️  Weather loaded: {len(weather_df):,} hours")
        
        # Monitor memory usage
        monitor_memory_usage()
        
        # Step 2: Compute MRT with parallel processing
        print("\n" + "="*40)
        print("STEP 2: COMPUTING MRT (PARALLEL)")
        print("="*40)
        
        from MRT import MRTCalculator, create_validation_period_filter, create_rectangular_grid
        
        # Create MRT calculator with context geometry
        mrt_calc = MRTCalculator(context_meshes=[model])
        mrt_calc.set_location_from_epw(epw_file)
        
        # Create simplified analysis grid that approximates Grasshopper approach
        
        grid_size = 10.0  # meters - smaller spacing for ~4000 points (match Grasshopper's 4158)
        
        print(f"🏗️  Generating simplified grid (Grasshopper-aligned spacing)")
        print(f"📐 Grid size: {grid_size}m")
        
        # Use exact user-specified bounds (adjusted to avoid edge artifacts)
        # Top left: (-4078.54, -1048.031), Bottom right: (-5069.535, -627)
        bounds_min = np.array([-5069.535, -1048.031])  # x_min, y_min
        bounds_max = np.array([-4078.54, -635.0])      # x_max, y_max (adjusted to -635 to avoid edge line)
        
        # Create ground-level grid at human height using exact bounds
        grid = create_rectangular_grid(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            grid_size=grid_size,
            z_height=1.5  # Human height for pedestrian analysis
        )
        
        print(f"🎯 User-specified bounds: [{bounds_min[0]:.1f}, {bounds_min[1]:.1f}] to [{bounds_max[0]:.1f}, {bounds_max[1]:.1f}]")
        print(f"📏 Grid area: {(bounds_max[0] - bounds_min[0]):.1f}m × {(bounds_max[1] - bounds_min[1]):.1f}m")
        
        print(f"🔢 Grid generated: {len(grid.points)} points at {grid_size}m spacing")
        
        # Debug: Show grid extent and first few actual positions
        points = np.array(grid.points)
        print(f"🔍 Grid extent: X=[{points[:,0].min():.1f}, {points[:,0].max():.1f}], Y=[{points[:,1].min():.1f}, {points[:,1].max():.1f}], Z={points[0,2]:.1f}")
        print(f"📍 First 3 computed positions: {points[:3].tolist()}")
        
        # Create analysis period and target hours based on user choice
        analysis_period, target_hours = create_analysis_period_and_hours(analysis_mode, target_hour)
        
        # Compute exposure (this uses parallel processing automatically)
        if analysis_mode == "full_day":
            print("🔍 Computing exposure for all 24 hours with parallel processing...")
            print(f"📊 Processing {len(grid.points):,} positions × {len(target_hours)} hours = {len(grid.points) * len(target_hours):,} calculations")
            print("⏱️  This may take several minutes for full day analysis...")
        else:
            print("🔍 Computing exposure with parallel processing...")
        
        try:
            exposure_results = mrt_calc.compute_exposure(
                positions=grid.points,
                analysis_period=analysis_period,
                target_hours=target_hours
            )
            
            # Monitor memory after exposure calculation
            print("💾 Memory after exposure calculation:")
            monitor_memory_usage()
            
            # Compute MRT
            if analysis_mode == "full_day":
                print("🌡️  Computing MRT for all 24 hours...")
            else:
                print("🌡️  Computing MRT...")
            
            mrt_results = mrt_calc.compute_mrt(
                epw_data=epw_data,
                exposure_results=exposure_results,
                analysis_period=analysis_period,
                target_hours=target_hours
            )
            
            # Clean up exposure results to free memory
            del exposure_results
            cleanup_memory()
            
        except Exception as e:
            print(f"❌ Error in MRT calculation: {e}")
            raise
        
        print(f"✅ MRT computed for {len(mrt_results)} positions")
        
        # Step 3: Compute UTCI
        print("\n" + "="*40)
        print("STEP 3: COMPUTING UTCI")
        print("="*40)
        
        from utci_calculator import UTCICalculator
        
        # Create UTCI calculator with weather data
        utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data)
        
        # Compute UTCI
        if analysis_mode == "full_day":
            print("🌡️  Computing UTCI for all 24 hours from MRT and weather data...")
        else:
            print("🌡️  Computing UTCI from MRT and weather data...")
        
        try:
            utci_results = utci_calc.compute_utci(
                mrt_results=mrt_results,
                analysis_period=analysis_period,
                target_hours=target_hours,
                show_progress=True
            )
            
            # Monitor memory after UTCI calculation
            print("💾 Memory after UTCI calculation:")
            monitor_memory_usage()
            
        except Exception as e:
            print(f"❌ Error in UTCI calculation: {e}")
            raise
        
        # Get summary statistics
        # UTCI results ready for export
        summary = {'comfort_assessment': 'Analysis complete'}
        
        # Create visualization based on analysis mode
        comparison_filename = create_visualization(
            analysis_mode=analysis_mode,
            model=model,
            utci_results=utci_results,
            validation_csv=validation_csv,
            grid_size=grid_size
        )
        
        # Auto-open in browser
        import webbrowser
        import os
        file_path = os.path.abspath(comparison_filename)
        webbrowser.open(f"file://{file_path}")
        
        # Export results with appropriate filename
        if analysis_mode == "single_hour":
            utci_output_path = f"utci_results_grid_{grid_size}m_simplified_70pct_hour_{target_hour:02d}.csv"
            print(f"💾 Exporting single hour results to: {utci_output_path}")
        else:
            utci_output_path = f"utci_results_grid_{grid_size}m_simplified_70pct_24hour.csv"
            print(f"💾 Exporting 24-hour results to: {utci_output_path}")
        
        utci_calc.to_csv(
            utci_results=utci_results,
            csv_path=utci_output_path,
            include_weather=True,
            include_comfort_categories=True
        )
        
        # Calculate actual UTCI statistics from results
        all_utci_values = []
        for pos_key, data in utci_results.items():
            if isinstance(data.get('utci'), (list, np.ndarray)):
                all_utci_values.extend(data['utci'])
            elif isinstance(data.get('utci'), (int, float)):
                all_utci_values.append(data['utci'])
        
        all_utci_values = np.array(all_utci_values)
        utci_min, utci_max = np.min(all_utci_values), np.max(all_utci_values)
        utci_mean = np.mean(all_utci_values)
        
        # Print clean summary
        if analysis_mode == "single_hour":
            print(f"\n🎉 COMPLETE: {len(utci_results)} positions analyzed for hour {target_hour:02d}:00")
        else:
            print(f"\n🎉 COMPLETE: {len(utci_results)} positions analyzed for 24 hours")
        
        print(f"🌡️  UTCI Range: {utci_min:.1f} to {utci_max:.1f} °C (mean: {utci_mean:.1f} °C)")
        print(f"💾 Results: {utci_output_path} | Visualization: {comparison_filename}")
        
        if analysis_mode == "full_day":
            print(f"🎬 Animated visualization with time slider available in: {comparison_filename}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
