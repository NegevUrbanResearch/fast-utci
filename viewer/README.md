# Fast-UTCI Web Viewer

A performant Three.js-based 3D viewer for UTCI (Universal Thermal Climate Index) thermal comfort analysis.

## Features

- **3D Visualization**: Interactive 3D model with UTCI point cloud overlay
- **Layer Support**: Buildings, trees, roads displayed with proper materials
- **Time Controls**: Animated slider for full-day (24-hour) analysis
- **Analytics**: Real-time comparison with Grasshopper validation data
- **Interactivity**: Click points to see UTCI values and thermal comfort categories
- **Screenshots**: Download visualization as PNG
- **Mobile Responsive**: Works on tablets and mobile devices

## Quick Start

1. **Generate Analysis Data**:
   ```bash
   # Quick automated analysis (single hour at 13:00)
   python quick_analysis.py
   
   # OR interactive analysis with full options
   python run_analysis.py
   ```
   This will create binary data files in `data/analyses/`

2. **Open Viewer**:
   - Open `viewer/index.html` in a web browser
   - Select an analysis to visualize
   - Interact with the 3D model using mouse/touch:
     - **Left click + drag**: Rotate view
     - **Right click + drag**: Pan
     - **Scroll**: Zoom
     - **Click point**: Show UTCI details

## Architecture

### Data Pipeline
```
quick_analysis.py/run_analysis.py → scripts/export_for_viewer.py → Binary files (.bin + .json)
```

### File Structure
```
/viewer/
  index.html              # Landing page with analysis selector
  viewer.html             # Main 3D viewer
  /js/
    ColorScale.js         # Ladybug UTCI color scale
    UTCIDataLoader.js     # Binary data parsing
    ModelLoader.js        # GLTF model loading
    UTCIRenderer.js       # Point cloud rendering
    TimeController.js     # Time slider for full-day analysis
    Analytics.js          # Grasshopper comparison
    ViewerApp.js          # Main application controller
  /css/
    viewer.css            # Styling

/data/
  /models/
    100.gltf              # 3D model
    model_layers.json     # Layer metadata
  /analyses/
    [analysis_id].bin     # Binary UTCI data
    [analysis_id].json    # Analysis metadata
  /validation/
    grasshopper_aug15_fullday.bin  # Validation data
```

## Binary Data Format

### Single Hour Analysis
```
[4 bytes: num_positions]
[num_positions × 12 bytes: positions (x,y,z as float32)]
[num_positions × 4 bytes: utci values (float32)]
```

### Full Day Analysis
```
[8 bytes: num_positions, num_hours (uint32, uint32)]
[num_positions × 12 bytes: positions (x,y,z as float32)]
[num_positions × 4 bytes: utci hour 0 (float32)]
[num_positions × 4 bytes: utci hour 1 (float32)]
...
[num_positions × 4 bytes: utci hour 23 (float32)]
```
## Technology Stack

- **Three.js**: 3D rendering (loaded from CDN)
- **ES6 Modules**: Modern JavaScript
- **No Build Step**: Pure vanilla JS, runs directly in browser
- **No Dependencies**: Everything loaded from CDN

## Browser Support

- Chrome/Edge 89+
- Firefox 87+
- Safari 15+
- Mobile Chrome/Safari

### Colors
Edit `ColorScale.js` to modify the UTCI color scale (currently using Ladybug Tools standard colors).

### Camera
Adjust initial camera position in `ViewerApp.js`:
```javascript
this.camera.position.set(-2000, 300, -400);
```

### Point Size
Modify point size in `UTCIRenderer.js`:
```javascript
size: 8  // pixels
```



