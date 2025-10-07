"""
Complete UTCI Workflow Demonstration for fast-utci.

This script demonstrates the full pipeline:
1. Load 3D model and weather data using reader.py
2. Compute MRT using MRT calculator with parallel processing
3. Compute UTCI using UTCI calculator
4. Visualize results with 3D heatmap viewer
5. Compare with Grasshopper validation data
"""

# =============================================================================
# CONFIGURATION - Easy to modify settings
# =============================================================================

# Grid spacing for analysis points (in meters)
# Smaller values = more detailed analysis but slower computation
# Larger values = faster computation but less detail
GRID_SIZE = 10.0  # meters

# =============================================================================

from pathlib import Path
import time
import numpy as np
import os
from typing import Tuple, List, Optional
import pandas as pd
import psutil
import gc
from fast_utci.colors import create_ladybug_utci_colorscale

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


def get_user_simplification_choice():
    """Get user choice for model simplification."""
    print("\n" + "="*50)
    print("MODEL SIMPLIFICATION")
    print("="*50)
    print("Model simplification can speed up calculations but may reduce accuracy.")
    print("1. Use original model (recommended for accuracy)")
    print("2. Simplify model to 70% (faster calculations)")
    print("="*50)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return False  # No simplification
        elif choice == "2":
            return True   # Simplify to 70%
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
    from fast_utci.mrt.period import create_analysis_period
    
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


def create_visualization(analysis_mode: str, enhanced_model, utci_results: dict, validation_csv: str, grid_size: float, simplify_model: bool = False, target_hours: List[int] = None) -> str:
    """
    Create appropriate visualization based on analysis mode.
    
    Args:
        analysis_mode: "single_hour" or "full_day"
        enhanced_model: EnhancedModel with layer information
        utci_results: UTCI calculation results
        validation_csv: Path to validation CSV
        grid_size: Grid spacing for filename
        simplify_model: Whether model was simplified
        target_hours: List of hours for full day analysis
        
    Returns:
        Filename of created visualization
    """
    from fast_utci.viewer import EnhancedUTCIViewer
    
    viewer = EnhancedUTCIViewer()
    
    if analysis_mode == "single_hour":
        # Single hour visualization
        print("📊 Creating single hour comparison with Grasshopper validation data...")
        
        comparison_fig = viewer.visualize_enhanced_utci_heatmap(
            enhanced_model=enhanced_model,
            utci_results=utci_results,
            title="UTCI Results: Enhanced Model vs Grasshopper Validation - Single Hour",
            analysis_type="single_hour",
            validation_csv=validation_csv
        )
        
        # Save the comparison as HTML file
        if simplify_model:
            comparison_filename = f"utci_comparison_grid_{grid_size}m_simplified_70pct_single_hour.html"
        else:
            comparison_filename = f"utci_comparison_grid_{grid_size}m_original_single_hour.html"
        
    else:  # full_day
        # Full day animated visualization with same detailed view as single hour
        print("📊 Creating animated 24-hour UTCI visualization with detailed view...")
        
        comparison_fig = viewer.visualize_enhanced_utci_heatmap(
            enhanced_model=enhanced_model,
            utci_results=utci_results,
            title="UTCI Results: 24-Hour Analysis - Enhanced Model with Time Controls",
            analysis_type="full_day",
            validation_csv=validation_csv,
            target_hours=target_hours
        )
        
        # Save the animated visualization as HTML file
        if simplify_model:
            comparison_filename = f"utci_comparison_grid_{grid_size}m_simplified_70pct_24hour_animated.html"
        else:
            comparison_filename = f"utci_comparison_grid_{grid_size}m_original_24hour_animated.html"
    
    comparison_fig.write_html(comparison_filename)
    print(f"💾 Visualization saved: {comparison_filename}")
    
    return comparison_filename


# Removed old animated visualization function - now using enhanced_viewer


def main():
    """Run the complete UTCI workflow demonstration."""
    
    print("=" * 60)
    print("FAST-UTCI COMPLETE WORKFLOW DEMONSTRATION")
    print("=" * 60)
    # Optional performance flags notice for easier benchmarking
    if os.getenv("FAST_UTCI_VECTORIZED_SOLAR"):
        print("⚙️  Using vectorized solar exposure path (FAST_UTCI_VECTORIZED_SOLAR)")
    if os.getenv("FAST_UTCI_INTERSECTOR"):
        print(f"⚙️  Ray intersector backend: {os.getenv('FAST_UTCI_INTERSECTOR')}")
    
    # Get user choices for analysis type and model simplification
    analysis_mode = get_user_analysis_choice()
    simplify_model = get_user_simplification_choice()
    
    if analysis_mode == "single_hour":
        target_hour = get_single_hour_input()
        print(f"✅ Selected: Single hour analysis for hour {target_hour:02d}:00")
    else:
        print("✅ Selected: Full day analysis (24 hours)")
        target_hour = None
    
    if simplify_model:
        print("✅ Selected: Model simplification to 70%")
    else:
        print("✅ Selected: Use original model (no simplification)")
    
    # Apply default performance flags for full-day runs (honor existing env overrides)
    if analysis_mode == "full_day":
        # Enable vectorized solar batching
        os.environ.setdefault("FAST_UTCI_VECTORIZED_SOLAR", "1")
        # Enable vectorized UTCI inside workers
        os.environ.setdefault("FAST_UTCI_VECTORIZED_UTCI", "1")
        # Prefer Embree backend and low quality suitable for occlusion queries
        os.environ.setdefault("FAST_UTCI_INTERSECTOR", "embree")
        os.environ.setdefault("FAST_UTCI_EMBREE_QUALITY", "low")
        os.environ.setdefault("FAST_UTCI_EMBREE_BUILD_BVH", "true")
        # Use fast boolean occlusion path when available
        os.environ.setdefault("FAST_UTCI_INTERSECTS_ANY", "1")
        # Batch across positions when pt_count==1 (safe, only engages in that case)
        os.environ.setdefault("FAST_UTCI_BATCH_POSITIONS", "1")
        # Reduce per-position payload in UTCI worker to cut serialization cost
        os.environ.setdefault("FAST_UTCI_INCLUDE_WEATHER_IN_RESULTS", "0")
        os.environ.setdefault("FAST_UTCI_INCLUDE_DATETIME_IN_RESULTS", "0")
        print("⚡ Optimizations enabled for full day: vectorized solar, Embree(low), intersects_any, batch positions")
    else:
        # Single-hour defaults: enable Embree + intersects_any; enable position batching (pt_count==1)
        os.environ.setdefault("FAST_UTCI_INTERSECTOR", "embree")
        os.environ.setdefault("FAST_UTCI_EMBREE_QUALITY", "low")
        os.environ.setdefault("FAST_UTCI_EMBREE_BUILD_BVH", "true")
        os.environ.setdefault("FAST_UTCI_INTERSECTS_ANY", "1")
        os.environ.setdefault("FAST_UTCI_BATCH_POSITIONS", "1")
        # Do not set FAST_UTCI_VECTORIZED_SOLAR by default for single hour
        print("⚡ Optimizations enabled for single hour: Embree(low), intersects_any, batch positions")

    # Validate analysis mode
    if not validate_analysis_mode(analysis_mode, target_hour):
        return 1
    
    # File paths
    model_file = "data/3d_models/100.gltf"
    epw_file = "data/weather/ISR_Beer.Sheva.401900_MSI.epw" 
    
    # Set validation CSV based on analysis mode
    if analysis_mode == "single_hour":
        validation_csv = "data/validation/15th_aug_13_14_MRT.csv"
    else:  # full_day
        validation_csv = "data/validation/15th_Aug_MRT.csv"
    
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
        
        from fast_utci.model_reader import read_project_data_enhanced
        t0 = time.perf_counter()
        enhanced_model, weather_df, epw_data = read_project_data_enhanced(model_file, epw_file, verbose=False)
        t1 = time.perf_counter()
        print(f"⏱️  Load time: {(t1-t0):.2f}s")
        model = enhanced_model.get_combined_mesh()  # Get combined mesh for MRT calculations
        
        print(f"📊 Enhanced model loaded: {len(model.vertices):,} vertices, {len(model.faces):,} faces")
        
        # Display layer information (consolidated)
        layer_info = enhanced_model.get_layer_info()
        layer_counts = {}
        for info in layer_info:
            layer_type = info['display_name']
            if layer_type not in layer_counts:
                layer_counts[layer_type] = {'count': 0, 'vertices': 0, 'faces': 0}
            layer_counts[layer_type]['count'] += 1
            layer_counts[layer_type]['vertices'] += info['vertices']
            layer_counts[layer_type]['faces'] += info['faces']
        
        print(f"🏗️  Model layers ({len(layer_info)} total):")
        for layer_type, counts in layer_counts.items():
            print(f"  - {layer_type}: {counts['count']} objects, {counts['vertices']:,} vertices, {counts['faces']:,} faces")
        
        # Apply model simplification if requested
        if simplify_model:
            print("🔧 Simplifying model to 70% for performance comparison...")
            import trimesh
            target_faces = int(len(model.faces) * 0.7)
            model = model.simplify_quadric_decimation(face_count=target_faces)
            print(f"📊 Simplified model: {len(model.vertices):,} vertices, {len(model.faces):,} faces")
            original_faces = len(model.faces) / 0.7  # Calculate original face count
            speedup = original_faces / len(model.faces)
            print(f"⚡ Expected ray casting speedup: ~{speedup:.1f}x faster")
        else:
            print("✅ Using original model without simplification")
        print(f"🌤️  Weather loaded: {len(weather_df):,} hours")
        
        # Monitor memory usage
        monitor_memory_usage()
        
        # Step 2: Compute MRT with parallel processing
        print("\n" + "="*40)
        print("STEP 2: COMPUTING MRT (PARALLEL)")
        print("="*40)
        
        from fast_utci.mrt import MRTCalculator, create_validation_period_filter, create_rectangular_grid
        
        # Create MRT calculator with context geometry
        mrt_calc = MRTCalculator(context_meshes=[model])
        # Report intersector backend if available
        if getattr(mrt_calc, 'mesh_context', None) is not None:
            backend = getattr(mrt_calc.mesh_context, 'backend_name', 'unknown')
            print(f"🧭 Ray intersector backend: {backend}")
        mrt_calc.set_location_from_epw(epw_file)
        
        # Create analysis grid using exact model bounds (no buffer)
        
        grid_size = GRID_SIZE  # Use configured grid size
        
        print(f"🏗️  Generating grid using exact model bounds (no buffer)")
        print(f"📐 Grid size: {grid_size}m")
        
        # Use exact model bounds as specified by user
        # Model bounds: x: -2470.81 to -1479.529, y: -619.8652 to -196.4804
        bounds_min = np.array([-2470.81, -619.8652])  # x_min, y_min
        bounds_max = np.array([-1479.529, -196.4804])  # x_max, y_max
        
        print(f"📊 Using exact model bounds: X=[{bounds_min[0]:.1f}, {bounds_max[0]:.1f}], Y=[{bounds_min[1]:.1f}, {bounds_max[1]:.1f}]")
        
        # Create ground-level grid at human height using exact bounds
        grid = create_rectangular_grid(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            grid_size=grid_size,
            z_height=1.5  # Human height for pedestrian analysis
        )
        
        print(f"🎯 Analysis bounds: [{bounds_min[0]:.1f}, {bounds_min[1]:.1f}] to [{bounds_max[0]:.1f}, {bounds_max[1]:.1f}]")
        print(f"📏 Grid area: {(bounds_max[0] - bounds_min[0]):.1f}m × {(bounds_max[1] - bounds_min[1]):.1f}m (exact model bounds)")
        print(f"🏢 Model coverage: 100% of grid area (no buffer)")
        
        print(f"🔢 Grid generated: {len(grid.points)} points at {grid_size}m spacing")
        
        # Show grid extent (simplified)
        points = np.array(grid.points)
        print(f"🔍 Grid extent: X=[{points[:,0].min():.1f}, {points[:,0].max():.1f}], Y=[{points[:,1].min():.1f}, {points[:,1].max():.1f}], Z={points[0,2]:.1f}")
        
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
            t2 = time.perf_counter()
            exposure_results = mrt_calc.compute_exposure(
                positions=grid.points,
                analysis_period=analysis_period,
                target_hours=target_hours
            )
            t3 = time.perf_counter()
            print(f"⏱️  Exposure compute time: {(t3-t2):.2f}s")
            
            # Monitor memory after exposure calculation
            print("💾 Memory after exposure calculation:")
            monitor_memory_usage()
            
            # Compute MRT
            if analysis_mode == "full_day":
                print("🌡️  Computing MRT for all 24 hours...")
            else:
                print("🌡️  Computing MRT...")
            
            t4 = time.perf_counter()
            mrt_results = mrt_calc.compute_mrt(
                epw_data=epw_data,
                exposure_results=exposure_results,
                analysis_period=analysis_period,
                target_hours=target_hours
            )
            t5 = time.perf_counter()
            print(f"⏱️  MRT compute time: {(t5-t4):.2f}s")
            
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
        
        from fast_utci.utci_calculator import UTCICalculator
        
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
            enhanced_model=enhanced_model,
            utci_results=utci_results,
            validation_csv=validation_csv,
            grid_size=grid_size,
            simplify_model=simplify_model,
            target_hours=target_hours
        )
        
        # Auto-open in browser
        import webbrowser
        file_path = os.path.abspath(comparison_filename)
        webbrowser.open(f"file://{file_path}")
        
        # Export results with appropriate filename
        if analysis_mode == "single_hour":
            if simplify_model:
                utci_output_path = f"utci_results_grid_{grid_size}m_simplified_70pct_hour_{target_hour:02d}.csv"
            else:
                utci_output_path = f"utci_results_grid_{grid_size}m_original_hour_{target_hour:02d}.csv"
            print(f"💾 Exporting single hour results to: {utci_output_path}")
        else:
            if simplify_model:
                utci_output_path = f"utci_results_grid_{grid_size}m_simplified_70pct_24hour.csv"
            else:
                utci_output_path = f"utci_results_grid_{grid_size}m_original_24hour.csv"
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
