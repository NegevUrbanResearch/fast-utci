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
    
    // Create buffer geometry
    const geometry = new THREE.BufferGeometry();
    
    // Calculate coordinate transformation to align UTCI data with model
    // The UTCI grid might be in a different coordinate system than the GLTF model
    const modelBox = new THREE.Box3().setFromObject(model);
    const modelCenter = new THREE.Vector3();
    modelBox.getCenter(modelCenter);
    const groundLevel = modelBox.min.y;  // Use bottom of model as ground level
    
    // Calculate UTCI grid center from metadata bounds
    // Note: UTCI data uses Z-up coordinate system, but Three.js uses Y-up
    // So we need to swap Y and Z when reading bounds
    const utciBounds = metadata.bounds;
    const utciCenter = new THREE.Vector3(
        (utciBounds.x_min + utciBounds.x_max) / 2,
        utciBounds.z || 0,  // Y in Three.js = Z in UTCI data (height)
        (utciBounds.y_min + utciBounds.y_max) / 2  // Z in Three.js = Y in UTCI data
    );
    
    // Calculate offset to center UTCI grid on model
    // Align UTCI grid to model center in X and Z, but place at ground level (Y=0)
    
    const offset = new THREE.Vector3(
        modelCenter.x - utciCenter.x,
        groundLevel - utciCenter.y,  // Place UTCI at ground level
        modelCenter.z + utciCenter.z  // Negate for 180° rotation
    );
    
    console.log(`[TRANSFORM] Model center: (${modelCenter.x.toFixed(2)}, ${modelCenter.y.toFixed(2)}, ${modelCenter.z.toFixed(2)})`);
    console.log(`[TRANSFORM] Model ground level: ${groundLevel.toFixed(2)}`);
    console.log(`[TRANSFORM] UTCI center (Y-up): (${utciCenter.x.toFixed(2)}, ${utciCenter.y.toFixed(2)}, ${utciCenter.z.toFixed(2)})`);
    console.log(`[TRANSFORM] Applying offset: (${offset.x.toFixed(2)}, ${offset.y.toFixed(2)}, ${offset.z.toFixed(2)})`);
    
    // Create position attribute with coordinate system transformation
    // UTCI data: X, Y, Z (Z-up) → Three.js: X, Y, Z (Y-up)
    // Swap Y and Z coordinates from data
    // Also negate Z to rotate 180 degrees horizontally to align with model
    // Add small vertical offset (0.2m) to prevent z-fighting/flickering with ground
    const VISUAL_OFFSET = 0.2;
    const positions = new Float32Array(data.positions.length);
    for (let i = 0; i < numPositions; i++) {
        positions[i * 3] = data.positions[i * 3] + offset.x;         // X stays X
        positions[i * 3 + 1] = data.positions[i * 3 + 2] + offset.y + VISUAL_OFFSET; // Y = Z from data (height) + offset
        positions[i * 3 + 2] = -(data.positions[i * 3 + 1]) + offset.z; // Z = -Y from data (180° rotation)
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
    points.name = 'UTCI_Points';
    
    console.log(`[OK] Point cloud created: ${numPositions} points for hour ${hourIndex}`);
    
    return points;
}

/**
 * Update point cloud colors for a different hour (full day analysis only)
 * @param {THREE.Points} pointCloud - Existing point cloud object
 * @param {object} analysis - Analysis data from UTCIDataLoader
 * @param {number} hourIndex - New hour index
 */
export function updatePointCloudColors(pointCloud, analysis, hourIndex) {
    const { data, metadata } = analysis;
    
    if (data.numHours === 1) {
        console.warn('[WARN] Cannot update colors for single hour analysis');
        return;
    }
    
    const numPositions = data.numPositions;
    const utciValues = data.utciByHour[hourIndex];
    const utciMin = metadata.utci_range.min;
    const utciMax = metadata.utci_range.max;
    
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
 * @returns {HTMLElement} Legend element
 */
export function createColorLegend(utciMin, utciMax) {
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
        min-width: 200px;
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
    
    const html = `
        <div style="font-weight: bold; margin-bottom: 12px; font-size: 15px;">UTCI Thermal Comfort</div>
        <div style="font-size: 12px; color: #666; margin-bottom: 15px;">
            Range: ${utciMin.toFixed(1)} - ${utciMax.toFixed(1)}°C
        </div>
        <div style="position: relative; display: flex; align-items: stretch;">
            <div style="width: 35px; height: 250px; background: linear-gradient(to bottom, ${steppedGradient}); border: 2px solid #666; border-radius: 5px; box-shadow: inset 0 0 5px rgba(0,0,0,0.1);"></div>
            <div style="position: relative; flex: 1; height: 250px; min-width: 65px;">
                ${labelsHTML}
            </div>
        </div>
    `;
    
    legend.innerHTML = html;
    return legend;
}
