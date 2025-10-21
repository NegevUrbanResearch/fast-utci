/**
 * UTCI Renderer for Point Cloud Visualization
 * 
 * Renders UTCI data as colored points in 3D space using Three.js BufferGeometry.
 */

import * as THREE from 'three';
import { mapUTCIToColor, LADYBUG_NUANCED_COLORS } from './ColorScale.js';

/**
 * Create UTCI point cloud geometry
 * @param {object} analysis - Analysis data from UTCIDataLoader
 * @param {THREE.Group} model - The loaded 3D model for coordinate alignment
 * @param {number} hourIndex - Hour index for full day analysis (default: 0)
 * @returns {THREE.Points} Points object with colored vertices
 */
export function createUTCIPointCloud(analysis, model, hourIndex = 0) {
    const { data, metadata } = analysis;
    const numPositions = data.numPositions;
    const coordinateSystem = metadata.coordinate_system || 'xy_ground';
    
    // Create buffer geometry
    const geometry = new THREE.BufferGeometry();
    
    // Calculate coordinate transformation to align UTCI data with model
    const modelBox = new THREE.Box3().setFromObject(model);
    const modelCenter = new THREE.Vector3();
    modelBox.getCenter(modelCenter);
    const groundLevel = modelBox.min.y;  // Use bottom of model as ground level
    
    console.log(`[UTCI] Coordinate system: ${coordinateSystem}`);
    
    // Create position attribute directly from data
    // Small vertical offset to prevent z-fighting with ground
    const VISUAL_OFFSET = 0.4;
    const positions = new Float32Array(data.positions.length);
    
    // Load positions as-is from data file
    for (let i = 0; i < numPositions; i++) {
        positions[i * 3] = data.positions[i * 3];
        positions[i * 3 + 1] = data.positions[i * 3 + 1];
        positions[i * 3 + 2] = data.positions[i * 3 + 2] + VISUAL_OFFSET;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Get UTCI values for this hour
    let utciValues;
    if (data.numHours === 1) {
        utciValues = data.utciValues;
    } else {
        utciValues = data.utciByHour[hourIndex];
    }
    
    // Create color attribute using dynamic colorscale
    const colors = new Float32Array(numPositions * 3);
    const utciMin = metadata.utci_range.min;
    const utciMax = metadata.utci_range.max;
    
    for (let i = 0; i < numPositions; i++) {
        const utci = utciValues[i];
        const color = mapUTCIToColor(utci, utciMin, utciMax);
        
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
    }
    
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    // Create material with improved rendering to prevent flickering
    const material = new THREE.PointsMaterial({
        size: 8,
        vertexColors: true,
        sizeAttenuation: false, // Keep size constant regardless of distance
        transparent: true,      // Enable transparency to fix z-fighting
        opacity: 0.95,          // Slightly transparent
        depthTest: true,        // Test depth to hide points behind objects
        depthWrite: false,      // Don't write to depth buffer to prevent flickering
        alphaTest: 0.1         // Discard fully transparent pixels
    });
    
    // Create points object
    const points = new THREE.Points(geometry, material);
    points.renderOrder = 2; // Render after most opaque meshes
    points.name = 'UTCI_Points';
    
    // Apply the same coordinate system transformation as the model
    // This ensures the UTCI grid aligns with the model
    if (coordinateSystem === 'xy_ground') {
        // Model uses Z-up (XY ground), rotate to Y-up for Three.js
        console.log('[TRANSFORM] Applying Z-up to Y-up rotation to UTCI points (-90° around X)');
        points.rotation.x = -Math.PI / 2;
    }
    
    console.log(`[OK] Point cloud created: ${numPositions} points for hour ${hourIndex}`);
    
    return points;
}

/**
 * Update point cloud colors for a different hour (full day analysis only)
 * @param {THREE.Points} pointCloud - Existing point cloud object
 * @param {object} analysis - Analysis data from UTCIDataLoader
 * @param {number} hourIndex - New hour index
 * @param {string} colorMode - 'normalized' (full day range) or 'discrete' (per-hour range)
 */
export function updatePointCloudColors(pointCloud, analysis, hourIndex, colorMode = 'normalized') {
    const { data, metadata } = analysis;
    
    if (data.numHours === 1) {
        console.warn('[WARN] Cannot update colors for single hour analysis');
        return;
    }
    
    const numPositions = data.numPositions;
    const utciValues = data.utciByHour[hourIndex];
    
    // Determine range based on color mode
    let utciMin, utciMax;
    if (colorMode === 'normalized') {
        // Use global range across all hours
        utciMin = metadata.utci_range.min;
        utciMax = metadata.utci_range.max;
    } else {
        // Use per-hour range for discrete coloring
        if (metadata.hour_statistics && metadata.hour_statistics[hourIndex]) {
            utciMin = metadata.hour_statistics[hourIndex].min;
            utciMax = metadata.hour_statistics[hourIndex].max;
        } else {
            // Fallback to global range if hour stats not available
            utciMin = metadata.utci_range.min;
            utciMax = metadata.utci_range.max;
        }
    }
    
    // Get color attribute
    const colorAttribute = pointCloud.geometry.getAttribute('color');
    
    // Update colors
    for (let i = 0; i < numPositions; i++) {
        const utci = utciValues[i];
        const color = mapUTCIToColor(utci, utciMin, utciMax);
        
        colorAttribute.setXYZ(i, color.r, color.g, color.b);
    }
    
    // Mark for update
    colorAttribute.needsUpdate = true;
}

/**
 * Create a raycaster for point picking
 * @param {THREE.Camera} camera - Three.js camera
 * @param {THREE.Vector2} mouse - Normalized mouse coordinates (-1 to 1)
 * @param {number} gridSize - Grid spacing in meters (for dynamic threshold)
 * @returns {THREE.Raycaster} Raycaster for intersection testing
 */
export function createRaycaster(camera, mouse, gridSize = 10.0) {
    const raycaster = new THREE.Raycaster();
    // Set threshold to half the grid size for accurate point picking
    // This allows hovering within half a grid spacing to select a point
    raycaster.params.Points.threshold = gridSize * 0.5;
    raycaster.setFromCamera(mouse, camera);
    return raycaster;
}

/**
 * Find nearest point to mouse cursor
 * @param {THREE.Points} pointCloud - Point cloud object
 * @param {THREE.Raycaster} raycaster - Raycaster from createRaycaster()
 * @returns {object|null} Intersection result or null if no hit
 */
export function findNearestPoint(pointCloud, raycaster) {
    const intersects = raycaster.intersectObject(pointCloud);
    
    if (intersects.length > 0) {
        const intersect = intersects[0];
        return {
            index: intersect.index,
            point: intersect.point,
            distance: intersect.distance
        };
    }
    
    return null;
}

/**
 * Create highlight sphere for selected point
 * @param {THREE.Vector3} position - Position to highlight
 * @param {number} radius - Sphere radius (default: 1.0)
 * @returns {THREE.Mesh} Highlight sphere mesh
 */
export function createHighlightSphere(position, radius = 1.0) {
    const geometry = new THREE.SphereGeometry(radius, 16, 16);
    const material = new THREE.MeshBasicMaterial({
        color: 0xffff00,
        transparent: true,
        opacity: 0.5,
        depthTest: false
    });
    
    const sphere = new THREE.Mesh(geometry, material);
    sphere.position.copy(position);
    sphere.name = 'Highlight_Sphere';
    
    return sphere;
}

/**
 * Create UTCI color legend as HTML with gradient bar showing actual data range
 * @param {number} utciMin - Minimum UTCI value  
 * @param {number} utciMax - Maximum UTCI value
 * @param {string} analysisType - 'single_hour' or 'full_day'
 * @returns {HTMLElement} Legend element
 */
export function createColorLegend(utciMin, utciMax, analysisType = 'single_hour') {
    const legend = document.createElement('div');
    legend.id = 'utci-legend';
    legend.style.cssText = `
        position: absolute;
        bottom: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 13px;
        width: auto;
        min-width: 0;
        z-index: 100;
    `;
    
    // Create stepped gradient from Ladybug colors (reversed for top-to-bottom display)
    const colors = [...LADYBUG_NUANCED_COLORS].reverse();
    const numColors = colors.length;
    const stepSize = 100 / numColors;
    
    // Build stepped gradient with hard transitions
    const gradientStops = [];
    for (let i = 0; i < numColors; i++) {
        const startPercent = (i * stepSize).toFixed(2);
        const endPercent = ((i + 1) * stepSize).toFixed(2);
        gradientStops.push(`${colors[i]} ${startPercent}%`);
        gradientStops.push(`${colors[i]} ${endPercent}%`);
    }
    const steppedGradient = gradientStops.join(', ');
    
    // Calculate temperature labels (5-7 evenly spaced points)
    const tempRange = utciMax - utciMin;
    const numLabels = 6;
    const tempStep = tempRange / (numLabels - 1);
    
    let labelsHTML = '';
    for (let i = 0; i < numLabels; i++) {
        const temp = utciMax - (i * tempStep);  // Start from max (top) to min (bottom)
        const position = (i / (numLabels - 1)) * 100;
        labelsHTML += `
            <div style="position: absolute; left: 8px; top: ${position}%; transform: translateY(-50%); font-size: 12px; font-weight: 500; white-space: nowrap;">
                ${temp.toFixed(1)}°C
            </div>
        `;
    }
    
    // Add switch control only for full day analysis
    let controlHTML = '';
    if (analysisType === 'full_day') {
        // Shoelace switch with inline labels; graceful fallback div kept minimal
        controlHTML = `
            <div style="margin-top: 12px; display: flex; align-items: center; justify-content: space-between; gap: 10px;">
                <div style="font-size: 12px; color: #333;">Color Scale</div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 11px; color: #555;">Full Day</span>
                    <sl-switch id="color-mode-switch" aria-label="Color scale mode" style="--width: 42px; --height: 22px; --thumb-size: 18px;"></sl-switch>
                    <span style="font-size: 11px; color: #555;">Per Hour</span>
                </div>
            </div>
        `;
    }
    
    const html = `
        <div style="font-weight: bold; margin-bottom: 12px; font-size: 15px;">UTCI</div>
        <div id="legend-range-text" style="font-size: 12px; color: #666; margin-bottom: 15px;">
            Range: ${utciMin.toFixed(1)} - ${utciMax.toFixed(1)}°C
        </div>
        <div style="position: relative; display: inline-flex; align-items: stretch; gap: 10px;">
            <div style="width: 35px; height: 250px; background: linear-gradient(to bottom, ${steppedGradient}); border: 2px solid #666; border-radius: 5px; box-shadow: inset 0 0 5px rgba(0,0,0,0.1);"></div>
            <div style="position: relative; height: 250px; width: 70px;">
                ${labelsHTML}
            </div>
        </div>
        ${controlHTML}
    `;
    
    legend.innerHTML = html;
    return legend;
}
