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

def main():
    """Run the complete UTCI workflow demonstration."""
    
    print("=" * 60)
    print("FAST-UTCI COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    # File paths
    model_file = "data/rec_model_no_curve.glb"
    epw_file = "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"  # Use TMYx weather file
    validation_csv = "data/15th_aug_13_14_MRT.csv"
    
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
        
        # Get analysis period (August 15th, hours 13-14)
        analysis_period, target_hours = create_validation_period_filter()
        print(f"📅 Analysis period: August 15th")
        print(f"⏰ Target hours: {target_hours}")
        
        # Compute exposure (this uses parallel processing automatically)
        print("🔍 Computing exposure with parallel processing...")
        exposure_results = mrt_calc.compute_exposure(
            positions=grid.points,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        # Compute MRT
        print("🌡️  Computing MRT...")
        mrt_results = mrt_calc.compute_mrt(
            epw_data=epw_data,
            exposure_results=exposure_results,
            analysis_period=analysis_period,
            target_hours=target_hours
        )
        
        print(f"✅ MRT computed for {len(mrt_results)} positions")
        
        # Step 3: Compute UTCI
        print("\n" + "="*40)
        print("STEP 3: COMPUTING UTCI")
        print("="*40)
        
        from utci_calculator import UTCICalculator
        
        # Create UTCI calculator with weather data
        utci_calc = UTCICalculator(weather_data=weather_df, epw_object=epw_data)
        
        # Compute UTCI
        print("🌡️  Computing UTCI from MRT and weather data...")
        utci_results = utci_calc.compute_utci(
            mrt_results=mrt_results,
            analysis_period=analysis_period,
            target_hours=target_hours,
            show_progress=True
        )
        
        # Get summary statistics
        # UTCI results ready for export
        summary = {'comfort_assessment': 'Analysis complete'}
        
        # Compare with Grasshopper validation
        from viewer import UTCIHeatmapViewer
        viewer = UTCIHeatmapViewer()
        
        print("📊 Creating comparison with Grasshopper validation data...")
        
        # Prepare positions for visualization
        first_utci_positions = []
        for i, (pos_key, data) in enumerate(list(utci_results.items())[:3]):
            pos = data['position']
            first_utci_positions.append(pos)
        
        comparison_fig = viewer.compare_with_grasshopper(
            model_mesh=model,
            utci_results=utci_results,
            grasshopper_csv=validation_csv,
            title="UTCI Results: Python (70% Simplified Model) vs Grasshopper Validation - TMYx Weather"
        )
        
        # Save the comparison as HTML file
        comparison_filename = f"utci_comparison_grid_{grid_size}m_simplified_70pct_tmyx.html"
        comparison_fig.write_html(comparison_filename)
        print(f"💾 Comparison saved: {comparison_filename}")
        
        # Auto-open in browser
        import webbrowser
        import os
        file_path = os.path.abspath(comparison_filename)
        webbrowser.open(f"file://{file_path}")
        
        # Export results
        utci_output_path = f"utci_results_grid_{grid_size}m_simplified_70pct_tmyx.csv"
        print(f"💾 Exporting results to: {utci_output_path}")
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
        print(f"\n🎉 COMPLETE: {len(utci_results)} positions analyzed")
        print(f"🌡️  UTCI Range: {utci_min:.1f} to {utci_max:.1f} °C (mean: {utci_mean:.1f} °C)")
        print(f"💾 Results: {utci_output_path} | Comparison: {comparison_filename}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
