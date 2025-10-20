/**
 * Model Loader for GLTF models and layer metadata
 * 
 * Loads 3D models and applies layer-based materials from metadata.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';

/**
 * Load layer metadata from JSON file
 * @param {string} metadataPath - Path to layer metadata JSON file
 * @returns {Promise<object>} Layer metadata object
 */
export async function loadLayerMetadata(metadataPath = '../data/models/model_layers.json') {
    const response = await fetch(metadataPath);
    if (!response.ok) {
        throw new Error(`Failed to load layer metadata: ${response.statusText}`);
    }
    return await response.json();
}

/**
 * Create material from layer metadata
 * @param {object} layer - Layer metadata object
 * @returns {THREE.Material} THREE.js material
 */
function createMaterialFromLayer(layer) {
    const isBuilding = layer.type === 'building';
    
    // Buildings use simpler, brighter MeshLambertMaterial
    if (isBuilding) {
        const mat = new THREE.MeshLambertMaterial({
            color: new THREE.Color(layer.color),
            emissive: new THREE.Color(0xffffff),  // Self-illumination for pure white
            emissiveIntensity: 0.3,                // Boost brightness
            side: THREE.DoubleSide,
            transparent: false,
            depthWrite: true
        });
        mat.polygonOffset = false; // buildings should occlude points cleanly
        return mat;
    }
    
    // Other layers use standard material with lighting
    const std = new THREE.MeshStandardMaterial({
        color: new THREE.Color(layer.color),
        opacity: layer.opacity,
        transparent: layer.opacity < 1.0,
        side: THREE.DoubleSide,
        roughness: 0.7,
        metalness: 0.1,
        depthWrite: layer.opacity > 0.5
    });
    // Slightly push base/ground away to avoid coplanar artifacts with UTCI overlay
    if (layer.type === 'base') {
        std.polygonOffset = true;
        std.polygonOffsetFactor = 1;
        std.polygonOffsetUnits = 1;
    }
    return std;
}

/**
 * Load GLTF model
 * @param {string} modelPath - Path to GLTF model file
 * @returns {Promise<THREE.Group>} Loaded model as THREE.Group
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
 * Detect material type based on mesh geometry properties
 * 
 * This replicates the Python logic from model_reader.py
 * 
 * @param {string} meshName - Name of the mesh
 * @param {THREE.Mesh} mesh - THREE.js mesh object
 * @returns {string} Material type
 */
function detectMaterialType(meshName, mesh) {
    const name = meshName.toLowerCase();
    
    // Check name-based detection first
    if (name.includes('building') || name.includes('wall') || name.includes('roof') || name.includes('facade')) {
        return 'building';
    } else if (name.includes('road') || name.includes('street') || name.includes('highway') || name.includes('pavement')) {
        return 'road';
    } else if (name.includes('sidewalk') || name.includes('footpath') || name.includes('walkway')) {
        return 'sidewalk';
    } else if (name.includes('tree') || name.includes('vegetation') || name.includes('plant') || name.includes('bush')) {
        return 'vegetation';
    } else if (name.includes('water') || name.includes('river') || name.includes('lake') || name.includes('pond')) {
        return 'water';
    }
    
    // Geometric analysis (replicating Python logic)
    const geometry = mesh.geometry;
    const position = geometry.getAttribute('position');
    
    if (!position) {
        return 'default';
    }
    
    // Calculate bounds
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    
    for (let i = 0; i < position.count; i++) {
        const x = position.getX(i);
        const y = position.getY(i);
        const z = position.getZ(i);
        
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        minZ = Math.min(minZ, z);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        maxZ = Math.max(maxZ, z);
    }
    
    // Calculate dimensions
    const boundsSize = new THREE.Vector3(maxX - minX, maxY - minY, maxZ - minZ);
    const height = boundsSize.z;
    const areaXY = boundsSize.x * boundsSize.y;
    // Note: minZ and maxZ are already defined above
    
    // Calculate aspect ratio (vertices per area)
    const aspectRatio = position.count / Math.max(areaXY, 1.0);
    
    console.log(`[GEOMETRY] ${meshName}: height=${height.toFixed(3)}m, area=${areaXY.toFixed(1)}m², aspect=${aspectRatio.toFixed(1)}, z=${minZ.toFixed(3)}-${maxZ.toFixed(3)}m`);
    
    // Apply the same logic as Python code
    
    // Base: truly flat, very large meshes (only the ground plane)
    if (height <= 0.001 && areaXY > 10000) {
        return 'base';
    }
    
    // Buildings: tall structures with significant height and reasonable area
    if (height > 2.0 && areaXY > 50) {
        return 'building';
    }
    
    // Roads: ANY linear elements (high aspect ratio) should be roads, regardless of height
    if (aspectRatio > 3.0) {
        return 'road';
    }
    
    // Trees: only truly elevated blob-like elements (low aspect ratio + elevated)
    if (minZ > 3.0 && aspectRatio < 2.0) {
        return 'vegetation';
    }
    
    // Elements just slightly above base should be roads/layout, not trees
    if (minZ > 0.1 && minZ <= 3.0) {
        return 'road';
    }
    
    // Elements at or below base level should never be trees
    if (minZ <= 0.1) {
        return 'road';
    }
    
    // Small volume elements = vegetation (only if truly elevated AND blob-like)
    if (minZ > 3.0 && aspectRatio < 2.0) {
        return 'vegetation';
    }
    
    // Default: if it has significant height and area, it's likely a building
    if (height > 1.0 && areaXY > 20) {
        return 'building';
    }
    
    return 'vegetation'; // Default fallback to vegetation
}

/**
 * Apply layer materials to model based on metadata
 * 
 * This function matches meshes to layer types based on geometric analysis
 * replicating the Python model_reader.py logic.
 * 
 * @param {THREE.Group} model - Loaded GLTF model
 * @param {object} layerMetadata - Layer metadata from loadLayerMetadata()
 * @returns {THREE.Group} Model with applied materials
 */
export function applyLayerMaterials(model, layerMetadata) {
    const materials = {};
    
    // Create materials for each layer type
    for (const layer of layerMetadata.layers) {
        materials[layer.type] = createMaterialFromLayer(layer);
    }
    
    // Track vegetation meshes for potential merging (performance optimization)
    const vegetationMeshes = [];
    let totalMeshCount = 0;
    
    // Apply materials to meshes
    model.traverse((child) => {
        totalMeshCount++;
        if (child.isMesh) {
            // Detect material type using geometric analysis
            const materialType = detectMaterialType(child.name, child);
            
            console.log(`[MATERIAL] ${child.name}: ${materialType}`);
            
            // Apply material
            if (materials[materialType]) {
                child.material = materials[materialType];
                console.log(`[APPLIED] ${materialType} material (${materials[materialType].color.getHexString()})`);
            } else {
                child.material = materials['default'];
                console.log(`[APPLIED] default material`);
            }
            
            // Add gray edges to buildings for better definition
            if (materialType === 'building') {
                const edges = new THREE.EdgesGeometry(child.geometry, 15); // 15 degree threshold
                const lineMaterial = new THREE.LineBasicMaterial({ 
                    color: 0x888888, // Medium gray
                    linewidth: 1,
                    transparent: false
                });
                const lineSegments = new THREE.LineSegments(edges, lineMaterial);
                lineSegments.name = `${child.name}_edges`;
                child.add(lineSegments);
            }
            
            // Performance optimization: disable shadows on vegetation (huge performance gain)
            if (materialType === 'vegetation') {
                child.castShadow = false;   // Vegetation doesn't cast shadows (performance)
                child.receiveShadow = true; // Still receives shadows
                vegetationMeshes.push(child);
            } else {
                // Enable shadows for other objects
                child.castShadow = true;
                child.receiveShadow = true;
            }
        }
    });
    
    // Merge vegetation geometry for massive performance boost
    if (vegetationMeshes.length > 0) {
        console.log(`[PERF] Merging ${vegetationMeshes.length} vegetation meshes for performance...`);
        
        const mergedGeometry = new THREE.BufferGeometry();
        const geometries = [];
        
        // Collect all vegetation geometries with world transforms
        vegetationMeshes.forEach(mesh => {
            const geometry = mesh.geometry.clone();
            mesh.updateWorldMatrix(true, false);
            geometry.applyMatrix4(mesh.matrixWorld); // Fixed: matrixWorld not worldMatrix
            geometries.push(geometry);
            
            // Remove original mesh from scene
            if (mesh.parent) {
                mesh.parent.remove(mesh);
            }
        });
        
        // Merge all geometries into one
        const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
        
        if (merged) {
            // Create single mesh for all vegetation
            const mergedMesh = new THREE.Mesh(merged, materials['vegetation']);
            mergedMesh.name = 'Vegetation_Merged';
            mergedMesh.castShadow = false;
            mergedMesh.receiveShadow = true;
            model.add(mergedMesh);
            
            console.log(`[PERF] Merged ${vegetationMeshes.length} meshes into 1 (${(merged.attributes.position.count / 1000).toFixed(1)}k vertices)`);
        } else {
            console.warn('[PERF] Failed to merge vegetation geometry');
        }
    }
    
    console.log('[OK] Layer materials applied');
    console.log(`[PERF] Total meshes: ${totalMeshCount}, Vegetation optimized: ${vegetationMeshes.length} -> 1 mesh`);
    return model;
}

/**
 * Load complete model with layer materials
 * @param {string} modelPath - Path to GLTF model file
 * @param {string} layerMetadataPath - Path to layer metadata JSON file
 * @returns {Promise<THREE.Group>} Model with applied materials
 */
export async function loadModelWithLayers(
    modelPath = '../data/3d_models/100.gltf',
    layerMetadataPath = '../data/models/model_layers.json'
) {
    console.log('[LOAD] Loading model and layer metadata...');
    
    // Load both in parallel
    const [model, layerMetadata] = await Promise.all([
        loadGLTFModel(modelPath),
        loadLayerMetadata(layerMetadataPath)
    ]);
    
    // Apply layer materials
    const modelWithMaterials = applyLayerMaterials(model, layerMetadata);
    
    return modelWithMaterials;
}

/**
 * Toggle visibility of model layers
 * @param {THREE.Group} model - THREE.js model
 * @param {string} layerType - Layer type to toggle (building, road, vegetation, base, water)
 * @param {boolean} visible - Whether to show or hide the layer
 */
export function toggleLayerVisibility(model, layerType, visible) {
    let toggleCount = 0;
    
    model.traverse((child) => {
        if (child.isMesh) {
            // Check if this mesh belongs to the specified layer type
            const materialType = detectMaterialType(child.name, child);
            if (materialType === layerType) {
                child.visible = visible;
                toggleCount++;
            }
        }
    });
    
    console.log(`[LAYER] ${visible ? 'Showing' : 'Hiding'} ${layerType}: ${toggleCount} meshes`);
}

/**
 * Calculate model bounds
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
