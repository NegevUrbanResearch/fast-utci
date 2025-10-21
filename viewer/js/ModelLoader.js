/**
 * Model Loader with Scene Graph Layer Detection
 * 
 * Loads GLTF models and applies materials based on layer names from the scene graph.
 * No geometric detection - uses actual layer names from modeling software.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import { LAYER_MATERIALS, LAYER_NAME_MAPPING } from '../config/layer-materials.js';

/**
 * Get layer name from scene graph by traversing parent chain
 * 
 * @param {THREE.Mesh} mesh - Mesh object to find layer for
 * @returns {string} Layer name from GLB scene graph
 */
function getLayerName(mesh) {
    let current = mesh;
    
    // Traverse up parent chain (max 20 levels to prevent infinite loops)
    for (let i = 0; i < 20; i++) {
        if (!current.parent) break;
        
        current = current.parent;
        const name = current.name;
        
        // Check if this is a layer node (meaningful name)
        if (name && 
            !name.match(/^\d+$/) &&              // Not just digits
            !name.startsWith('GLTF') &&          // Not auto-generated GLTF name
            !name.startsWith('Layer_') &&        // Ignore generic Layer_XX names
            name !== 'Scene' &&                  // Not root scene
            name !== '') {
            return name;  // Found the layer name!
        }
    }
    
    return 'unknown';
}

/**
 * Map actual GLB layer name to standard material type
 * 
 * @param {string} layerName - Layer name from scene graph
 * @returns {string} Standard layer type (building, road, vegetation, etc.)
 */
function mapLayerNameToType(layerName) {
    const nameLower = layerName.toLowerCase();
    
    // Try exact match first
    if (LAYER_NAME_MAPPING[nameLower]) {
        return LAYER_NAME_MAPPING[nameLower];
    }
    
    // Fallback to substring matching
    for (const [key, type] of Object.entries(LAYER_NAME_MAPPING)) {
        if (nameLower.includes(key)) {
            return type;
        }
    }
    
    return 'default';
}

/**
 * Detect layer type from geometry when scene graph name is unknown
 * Uses simple geometric heuristics to identify base layers
 * 
 * @param {THREE.Mesh} mesh - Mesh to analyze
 * @returns {string} Detected layer type
 */
function detectLayerTypeFromGeometry(mesh) {
    const geometry = mesh.geometry;
    
    // Compute bounding box if not already computed
    if (!geometry.boundingBox) {
        geometry.computeBoundingBox();
    }
    
    const bbox = geometry.boundingBox;
    const size = new THREE.Vector3();
    bbox.getSize(size);
    
    const height = size.z;
    const areaXY = size.x * size.y;
    
    // Base layer heuristic: truly flat (< 1cm) and large area (> 10000 m²)
    if (height < 0.01 && areaXY > 10000) {
        console.log(`[DETECT] Mesh with unknown name detected as base layer (height: ${height.toFixed(3)}m, area: ${areaXY.toFixed(1)}m²)`);
        return 'base';
    }
    
    // Default to 'default' type for other unknown meshes
    return 'default';
}

/**
 * Create Three.js material from layer configuration
 * 
 * @param {string} layerType - Standard layer type
 * @returns {THREE.Material} Configured material
 */
function createMaterialFromConfig(layerType) {
    const config = LAYER_MATERIALS[layerType] || LAYER_MATERIALS.default;
    
    // Buildings use Lambert material for performance
    if (config.materialType === 'lambert') {
        const mat = new THREE.MeshLambertMaterial({
            color: new THREE.Color(config.color),
            emissive: config.emissive ? new THREE.Color(config.emissive) : undefined,
            emissiveIntensity: config.emissiveIntensity || 0,
            side: THREE.DoubleSide,
            transparent: config.opacity < 1.0,
            opacity: config.opacity,
            depthWrite: true
        });
        mat.polygonOffset = false;  // Buildings should occlude points cleanly
        return mat;
    }
    
    // Other layers use Standard material
    const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(config.color),
        opacity: config.opacity,
        transparent: config.opacity < 1.0,
        side: THREE.DoubleSide,
        roughness: 0.7,
        metalness: 0.1,
        depthWrite: config.opacity > 0.5
    });
    
    // Apply polygon offset for base layer (prevents z-fighting with UTCI overlay)
    if (config.polygonOffset) {
        mat.polygonOffset = true;
        mat.polygonOffsetFactor = 1;
        mat.polygonOffsetUnits = 1;
    }
    
    return mat;
}

/**
 * Apply layer materials to loaded model
 * 
 * Traverses scene graph to find layer names and applies appropriate materials.
 * Also groups meshes by layer type for performance optimization.
 * 
 * @param {THREE.Group} model - Loaded GLTF model
 * @returns {THREE.Group} Model with applied materials
 */
export function applyLayerMaterials(model) {
    const meshesByLayer = {};  // Group meshes by layer type for merging
    const layerStats = {};
    const itemsToRemove = [];  // Track non-mesh items to remove
    
    model.traverse((child) => {
        // Remove lines/curves that aren't needed (not building edges we add)
        if (child.isLine || child.isLineSegments) {
            if (!child.name.includes('_edges')) {
                itemsToRemove.push(child);
            }
            return;
        }
        
        if (child.isMesh) {
            // Extract layer name from scene graph
            const layerName = getLayerName(child);
            let layerType = mapLayerNameToType(layerName);
            
            // Fallback: If layer name is 'unknown', try geometric detection for base layer
            if (layerName === 'unknown') {
                layerType = detectLayerTypeFromGeometry(child);
            }
            
            // Apply material based on layer type
            child.material = createMaterialFromConfig(layerType);
            
            // Store layer info in userData for later use (visibility toggles, etc.)
            child.userData.layerType = layerType;
            child.userData.layerName = layerName;
            
            // Track layer statistics
            if (!layerStats[layerType]) {
                layerStats[layerType] = { count: 0, layerName: layerName };
            }
            layerStats[layerType].count++;
            
            // Group meshes by layer for merging
            if (!meshesByLayer[layerType]) {
                meshesByLayer[layerType] = [];
            }
            meshesByLayer[layerType].push(child);
            
            // Shadow settings (all layers cast/receive shadows initially)
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
    
    // Remove unwanted lines/curves
    itemsToRemove.forEach(item => {
        if (item.parent) {
            item.parent.remove(item);
        }
    });
    if (itemsToRemove.length > 0) {
        console.log(`[FILTER] Removed ${itemsToRemove.length} line/curve objects`);
    }
    
    // Print layer summary
    console.log('[LAYERS] Discovered layers:');
    for (const [layerType, stats] of Object.entries(layerStats)) {
        console.log(`  ${layerType}: ${stats.count} meshes (from '${stats.layerName}')`);
    }
    
    // Debug: List all unique layer names found
    const uniqueLayerNames = new Set();
    for (const stats of Object.values(layerStats)) {
        uniqueLayerNames.add(stats.layerName);
    }
    console.log(`[DEBUG] Unique layer names in scene graph: ${[...uniqueLayerNames].join(', ')}`);
    console.log(`[DEBUG] Layer types that will be available: ${Object.keys(layerStats).join(', ')}`);
    
    // Merge geometries by layer type for massive performance improvement
    console.log('[PERF] Merging geometries by layer type...');
    for (const [layerType, meshes] of Object.entries(meshesByLayer)) {
        if (meshes.length > 1) {
            mergeLayerMeshes(model, layerType, meshes);
        }
    }
    
    return model;
}

/**
 * Merge layer meshes into a single geometry for massive performance improvement
 * 
 * @param {THREE.Group} model - Model to add merged mesh to
 * @param {string} layerType - Type of layer being merged
 * @param {Array<THREE.Mesh>} meshes - Array of meshes to merge
 */
function mergeLayerMeshes(model, layerType, meshes) {
    const geometries = [];
    
    console.log(`[PERF] Merging ${meshes.length} ${layerType} meshes...`);
    
    // Collect all geometries with world transforms
    meshes.forEach(mesh => {
        const geometry = mesh.geometry.clone();
        mesh.updateWorldMatrix(true, false);
        geometry.applyMatrix4(mesh.matrixWorld);
        geometries.push(geometry);
        
        // Remove original mesh from scene
        if (mesh.parent) {
            mesh.parent.remove(mesh);
        }
    });
    
    // Merge all geometries into one
    const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
    
    if (merged) {
        // Create single mesh for this layer
        const material = createMaterialFromConfig(layerType);
        const mergedMesh = new THREE.Mesh(merged, material);
        mergedMesh.name = `${layerType}_merged`;
        mergedMesh.userData.layerType = layerType;
        mergedMesh.userData.layerName = layerType;
        
        // Shadow settings based on layer type
        if (layerType === 'vegetation') {
            mergedMesh.castShadow = false;   // Vegetation doesn't cast shadows (performance)
            mergedMesh.receiveShadow = true;
        } else {
            mergedMesh.castShadow = true;
            mergedMesh.receiveShadow = true;
        }
        
        model.add(mergedMesh);
        
        const vertexCount = (merged.attributes.position.count / 1000).toFixed(1);
        console.log(`[PERF] ${layerType}: ${meshes.length} meshes -> 1 mesh (${vertexCount}k vertices)`);
        
        // Add building edges after merging for better performance
        if (layerType === 'building') {
            const edges = new THREE.EdgesGeometry(merged, 15);  // 15 degree threshold
            const lineMaterial = new THREE.LineBasicMaterial({ 
                color: 0x888888,  // Medium gray
                linewidth: 1
            });
            const lineSegments = new THREE.LineSegments(edges, lineMaterial);
            lineSegments.name = `${layerType}_edges`;
            mergedMesh.add(lineSegments);
            console.log(`[PERF] Added building edges to merged geometry`);
        }
    } else {
        console.warn(`[PERF] Failed to merge ${layerType} geometry`);
    }
}

/**
 * Load GLTF model
 * 
 * @param {string} modelPath - Path to GLTF model file
 * @returns {Promise<THREE.Group>} Loaded model
 */
export async function loadGLTFModel(modelPath) {
    return new Promise((resolve, reject) => {
        const loader = new GLTFLoader();
        
        loader.load(
            modelPath,
            (gltf) => {
                console.log('[OK] GLTF model loaded');
                resolve(gltf.scene);
            },
            (progress) => {
                const percent = (progress.loaded / progress.total * 100).toFixed(1);
                console.log(`[LOAD] Model loading: ${percent}%`);
            },
            (error) => {
                console.error('[ERROR] Failed to load GLTF model:', error);
                reject(error);
            }
        );
    });
}

/**
 * Load model with layer materials applied
 * 
 * @param {string} modelPath - Path to GLTF model file
 * @param {string} coordinateSystem - Coordinate system ('xy_ground' or 'xz_ground')
 * @returns {Promise<THREE.Group>} Model with applied materials and coordinate transform
 */
export async function loadModelWithLayers(
    modelPath = '../data/3d_models/100_test.glb',
    coordinateSystem = 'xy_ground'
) {
    console.log('[LOAD] Loading model...');
    console.log(`[INFO] Model path: ${modelPath}`);
    console.log(`[INFO] Coordinate system: ${coordinateSystem}`);
    
    // Load GLTF model
    const model = await loadGLTFModel(modelPath);
    
    // Apply layer materials based on scene graph
    const modelWithMaterials = applyLayerMaterials(model);
    
    // Apply coordinate transformation if needed
    // Three.js uses Y-up by default, so models with Z-up need rotation
    if (coordinateSystem === 'xy_ground') {
        // Model uses Z-up (XY is ground plane)
        // Rotate -90 degrees around X axis to convert Z-up to Y-up
        console.log('[TRANSFORM] Applying Z-up to Y-up rotation (-90° around X)');
        modelWithMaterials.rotation.x = -Math.PI / 2;
    }
    
    console.log('[OK] Model loaded with layer materials');
    return modelWithMaterials;
}

/**
 * Calculate model bounds
 * 
 * @param {THREE.Group} model - THREE.js model
 * @returns {THREE.Box3} Bounding box
 */
export function calculateModelBounds(model) {
    const box = new THREE.Box3().setFromObject(model);
    console.log('[INFO] Model bounds:', {
        min: box.min,
        max: box.max,
        size: box.getSize(new THREE.Vector3())
    });
    return box;
}
