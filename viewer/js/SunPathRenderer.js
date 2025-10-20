/**
 * Sun Path Renderer for UTCI Viewer
 * 
 * Renders the sun's path across the sky for a given date, showing:
 * - Arc curve representing the sun's trajectory
 * - Hour markers at each hour position
 * - Current sun position indicator
 */

import * as THREE from 'three';

/**
 * Create sun path visualization
 * @param {object} metadata - Analysis metadata with sun_positions and location
 * @param {THREE.Vector3} modelCenter - Center of the 3D model for positioning
 * @param {number} modelSize - Size of model for scaling
 * @returns {THREE.Group} Sun path group object
 */
export function createSunPath(metadata, modelCenter, modelSize) {
    const group = new THREE.Group();
    group.name = 'SunPath';
    group.visible = false; // Hidden by default
    
    const sunPositions = metadata.sun_positions;
    if (!sunPositions || sunPositions.length === 0) {
        console.warn('[SUNPATH] No sun position data available');
        return group;
    }
    
    const scale = modelSize * 0.6; // Sun path 0.6x model size for better visibility
    
    // Convert sun positions to 3D coordinates
    const points = sunPositions.map(sun => {
        if (!sun.is_up) {
            // Below horizon - calculate position from altitude/azimuth
            const alt = Math.max(sun.altitude, -90) * Math.PI / 180;
            const azi = sun.azimuth * Math.PI / 180;
            
            const x = Math.sin(azi) * Math.cos(alt) * scale;
            const y = Math.sin(alt) * scale;
            const z = Math.cos(azi) * Math.cos(alt) * scale;
            
            return new THREE.Vector3(x, y, z).add(modelCenter);
        } else {
            // Use pre-calculated vector from Python, scaled and transformed to Three.js coords
            // Python coords: X=East, Y=North, Z=Up
            // Three.js coords: X=East, Y=Up, Z=South (need to negate Y to get South)
            return new THREE.Vector3(
                sun.vector[0] * scale,      // X stays X (East)
                sun.vector[2] * scale,      // Y = Z from Python (Up)
                -sun.vector[1] * scale      // Z = -Y from Python (South)
            ).add(modelCenter);
        }
    });
    
    // Create arc curve (closed curve through all 24 hours)
    const curve = new THREE.CatmullRomCurve3(points, true);
    const arcGeometry = new THREE.TubeGeometry(curve, 64, 2, 8, false);
    const arcMaterial = new THREE.MeshBasicMaterial({ 
        color: 0xffaa00,
        transparent: true,
        opacity: 0.6
    });
    const arc = new THREE.Mesh(arcGeometry, arcMaterial);
    arc.name = 'sun_arc';
    group.add(arc);
    
    // Create hour markers
    sunPositions.forEach((sun, hour) => {
        const markerGeometry = new THREE.SphereGeometry(4, 16, 16);
        const markerMaterial = new THREE.MeshBasicMaterial({
            color: sun.is_up ? 0xffdd00 : 0x666666,
            transparent: true,
            opacity: sun.is_up ? 0.9 : 0.4
        });
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.copy(points[hour]);
        marker.userData = { hour, sunData: sun };
        marker.name = `sun_marker_${hour}`;
        group.add(marker);
    });
    
    console.log(`[SUNPATH] Created sun path with ${sunPositions.length} hour markers`);
    
    return group;
}

/**
 * Update current sun indicator
 * @param {THREE.Group} sunPathGroup - Sun path group
 * @param {number} currentHour - Current hour index (0-23)
 */
export function updateCurrentSunIndicator(sunPathGroup, currentHour) {
    if (!sunPathGroup) return;
    
    // Reset all markers to default state
    sunPathGroup.children.forEach(child => {
        if (child.name.startsWith('sun_marker_')) {
            const hour = child.userData.hour;
            const isUp = child.userData.sunData.is_up;
            child.material.color.setHex(isUp ? 0xffdd00 : 0x666666);
            child.scale.set(1, 1, 1);
        }
    });
    
    // Highlight current hour
    const currentMarker = sunPathGroup.getObjectByName(`sun_marker_${currentHour}`);
    if (currentMarker) {
        currentMarker.material.color.setHex(0xff3300); // Bright red-orange
        currentMarker.scale.set(4, 4, 4); // Much larger for visibility
        currentMarker.material.opacity = 1.0; // Full opacity
    }
}

/**
 * Create raycaster for sun marker picking
 * @param {THREE.Camera} camera - Three.js camera
 * @param {THREE.Vector2} mouse - Normalized mouse coordinates
 * @returns {THREE.Raycaster} Raycaster for intersection testing
 */
export function createSunMarkerRaycaster(camera, mouse) {
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 10;
    raycaster.setFromCamera(mouse, camera);
    return raycaster;
}

/**
 * Find clicked sun marker
 * @param {THREE.Group} sunPathGroup - Sun path group
 * @param {THREE.Raycaster} raycaster - Raycaster from createSunMarkerRaycaster()
 * @returns {object|null} User data of clicked marker or null
 */
export function findClickedSunMarker(sunPathGroup, raycaster) {
    if (!sunPathGroup) return null;
    
    const markers = sunPathGroup.children.filter(c => c.name.startsWith('sun_marker_'));
    const intersects = raycaster.intersectObjects(markers);
    
    if (intersects.length > 0) {
        return intersects[0].object.userData;
    }
    return null;
}

