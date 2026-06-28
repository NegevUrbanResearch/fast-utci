/**
 * Model Cache Service
 * 
 * Caches loaded GLTF models to avoid reloading when switching between scenarios.
 * Uses LRU eviction policy to limit memory usage.
 * Properly disposes of Three.js resources when models are evicted.
 */

import { LRUCache } from './lruCache';
import * as THREE from 'three';

function disposeTexture(value: unknown): void {
	if (value instanceof THREE.Texture) {
		value.dispose();
	}
}

export interface CachedModel {
	/** The loaded GLTF scene group */
	scene: THREE.Group;
	/** Timestamp when model was loaded */
	loadedAt: number;
}

// Create LRU cache with disposal callback
// Cache up to 5 models (reasonable for scenario switching)
const modelCache = new LRUCache<CachedModel>({
	maxSize: 5,
	onEvict: (key: string, value: CachedModel) => {
		console.log(`[MODEL CACHE] Evicting model: ${key}`);
		disposeModel(value.scene);
	}
});

/**
 * Dispose of a Three.js model and all its resources
 * Recursively disposes geometries, materials, and textures
 */
function disposeModel(object: THREE.Object3D): void {
	object.traverse((child) => {
		// Dispose geometries
		if (child instanceof THREE.Mesh) {
			if (child.geometry) {
				child.geometry.dispose();
			}

			// Dispose materials
			if (child.material) {
				if (Array.isArray(child.material)) {
					child.material.forEach((material) => disposeMaterial(material));
				} else {
					disposeMaterial(child.material);
				}
			}
		}

		// Dispose line segments
		if (child instanceof THREE.LineSegments) {
			if (child.geometry) {
				child.geometry.dispose();
			}
			if (child.material) {
				disposeMaterial(child.material as THREE.Material);
			}
		}
	});
}

/**
 * Dispose of a Three.js material and its textures
 */
function disposeMaterial(material: THREE.Material): void {
	const materialWithTextures = material as THREE.Material & {
		map?: unknown;
		lightMap?: unknown;
		bumpMap?: unknown;
		normalMap?: unknown;
		specularMap?: unknown;
		envMap?: unknown;
	};

	// Dispose textures
	disposeTexture(materialWithTextures.map);
	disposeTexture(materialWithTextures.lightMap);
	disposeTexture(materialWithTextures.bumpMap);
	disposeTexture(materialWithTextures.normalMap);
	disposeTexture(materialWithTextures.specularMap);
	disposeTexture(materialWithTextures.envMap);

	// Dispose material itself
	material.dispose();
}

/**
 * Get a model from the cache
 * Returns undefined if model is not cached
 * 
 * @param modelPath - Path to the model file
 * @returns Cached model or undefined
 */
export function getCachedModel(modelPath: string): CachedModel | undefined {
	const cached = modelCache.get(modelPath);
	if (cached) {
		console.log(`[MODEL CACHE] Cache hit: ${modelPath}`);
	}
	return cached;
}

/**
 * Store a model in the cache
 * If cache is full, least recently used model will be evicted and disposed
 * 
 * @param modelPath - Path to the model file
 * @param scene - The loaded GLTF scene
 */
export function cacheModel(modelPath: string, scene: THREE.Group): void {
	console.log(`[MODEL CACHE] Caching model: ${modelPath}`);
	modelCache.set(modelPath, {
		scene,
		loadedAt: Date.now()
	});
}

/**
 * Check if a model is cached
 * 
 * @param modelPath - Path to the model file
 * @returns True if model is in cache
 */
export function hasModelInCache(modelPath: string): boolean {
	return modelCache.has(modelPath);
}

/**
 * Remove a specific model from the cache and dispose it
 * 
 * @param modelPath - Path to the model file
 */
export function evictModel(modelPath: string): void {
	modelCache.delete(modelPath);
}

/**
 * Clear all cached models and dispose them
 */
export function clearModelCache(): void {
	console.log('[MODEL CACHE] Clearing all cached models');
	modelCache.clear();
}

/**
 * Get cache statistics
 */
export function getModelCacheStats(): {
	size: number;
	keys: string[];
} {
	return {
		size: modelCache.size,
		keys: modelCache.keys()
	};
}

