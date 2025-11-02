/**
 * Material Pool Service
 * 
 * Singleton service that creates and reuses Three.js materials for layer types.
 * This improves performance by avoiding material recreation and ensures
 * consistent material instances across the application.
 */

import * as THREE from 'three';
import { LAYER_MATERIALS } from '$lib/types/layerMaterials';
import type { LayerMaterialConfig } from '$lib/types/layers';

// Singleton material pool
const materialPool = new Map<string, THREE.Material>();

/**
 * Get a material for a layer type, creating it if it doesn't exist or reusing from pool
 * 
 * @param layerType - Standard layer type (building, vegetation, etc.)
 * @returns Three.js material instance
 */
export function getMaterial(layerType: string): THREE.Material {
	// Check if material already exists in pool
	if (materialPool.has(layerType)) {
		return materialPool.get(layerType)!;
	}

	// Get config for this layer type, fallback to default
	const config: LayerMaterialConfig = LAYER_MATERIALS[layerType] || LAYER_MATERIALS.default;

	// Create material based on materialType
	let material: THREE.Material;

	if (config.materialType === 'lambert') {
		// Buildings use Lambert material for performance
		const mat = new THREE.MeshLambertMaterial({
			color: new THREE.Color(config.color),
			emissive: config.emissive ? new THREE.Color(config.emissive) : undefined,
			emissiveIntensity: config.emissiveIntensity || 0,
			side: THREE.DoubleSide,
			transparent: config.opacity < 1.0,
			opacity: config.opacity,
			depthWrite: true
		});
		mat.polygonOffset = false; // Buildings should occlude points cleanly
		material = mat;
	} else {
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

		material = mat;
	}

	// Store in pool for reuse
	materialPool.set(layerType, material);
	return material;
}

/**
 * Clear all materials from the pool (useful for testing or cleanup)
 */
export function clearMaterialPool(): void {
	materialPool.forEach((material) => {
		material.dispose();
	});
	materialPool.clear();
}

