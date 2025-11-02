import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import {
	getCachedModel,
	cacheModel,
	hasModelInCache,
	evictModel,
	clearModelCache,
	getModelCacheStats
} from '$lib/services/modelCacheService';

describe('Model Cache Service', () => {
	beforeEach(() => {
		// Clear cache before each test
		clearModelCache();
	});

	describe('Basic caching operations', () => {
		it('should cache and retrieve a model', () => {
			const scene = new THREE.Group();
			scene.name = 'TestModel';

			cacheModel('/models/test.glb', scene);

			const cached = getCachedModel('/models/test.glb');
			expect(cached).toBeDefined();
			expect(cached?.scene).toBe(scene);
			expect(cached?.scene.name).toBe('TestModel');
		});

		it('should return undefined for uncached model', () => {
			const cached = getCachedModel('/models/nonexistent.glb');
			expect(cached).toBeUndefined();
		});

		it('should check if model is in cache', () => {
			const scene = new THREE.Group();
			cacheModel('/models/test.glb', scene);

			expect(hasModelInCache('/models/test.glb')).toBe(true);
			expect(hasModelInCache('/models/other.glb')).toBe(false);
		});

		it('should include load timestamp', () => {
			const scene = new THREE.Group();
			const beforeTime = Date.now();

			cacheModel('/models/test.glb', scene);

			const cached = getCachedModel('/models/test.glb');
			expect(cached?.loadedAt).toBeGreaterThanOrEqual(beforeTime);
			expect(cached?.loadedAt).toBeLessThanOrEqual(Date.now());
		});
	});

	describe('Cache eviction', () => {
		it('should evict least recently used model when capacity is reached', () => {
			// Create 6 models (cache max is 5)
			const models = Array.from({ length: 6 }, (_, i) => {
				const scene = new THREE.Group();
				scene.name = `Model${i + 1}`;
				return scene;
			});

			// Cache 5 models
			for (let i = 0; i < 5; i++) {
				cacheModel(`/models/model${i + 1}.glb`, models[i]);
			}

			expect(getModelCacheStats().size).toBe(5);
			expect(hasModelInCache('/models/model1.glb')).toBe(true);

			// Cache 6th model, should evict model1
			cacheModel('/models/model6.glb', models[5]);

			expect(getModelCacheStats().size).toBe(5);
			expect(hasModelInCache('/models/model1.glb')).toBe(false);
			expect(hasModelInCache('/models/model6.glb')).toBe(true);
		});

		it('should update LRU order when model is accessed', () => {
			const models = Array.from({ length: 6 }, (_, i) => {
				const scene = new THREE.Group();
				scene.name = `Model${i + 1}`;
				return scene;
			});

			// Cache 5 models
			for (let i = 0; i < 5; i++) {
				cacheModel(`/models/model${i + 1}.glb`, models[i]);
			}

			// Access model1 to make it most recently used
			getCachedModel('/models/model1.glb');

			// Cache 6th model, should now evict model2 instead of model1
			cacheModel('/models/model6.glb', models[5]);

			expect(hasModelInCache('/models/model1.glb')).toBe(true);
			expect(hasModelInCache('/models/model2.glb')).toBe(false);
		});
	});

	describe('Manual eviction', () => {
		it('should evict specific model', () => {
			const scene = new THREE.Group();
			cacheModel('/models/test.glb', scene);

			expect(hasModelInCache('/models/test.glb')).toBe(true);

			evictModel('/models/test.glb');

			expect(hasModelInCache('/models/test.glb')).toBe(false);
		});

		it('should clear all cached models', () => {
			const scene1 = new THREE.Group();
			const scene2 = new THREE.Group();
			const scene3 = new THREE.Group();

			cacheModel('/models/model1.glb', scene1);
			cacheModel('/models/model2.glb', scene2);
			cacheModel('/models/model3.glb', scene3);

			expect(getModelCacheStats().size).toBe(3);

			clearModelCache();

			expect(getModelCacheStats().size).toBe(0);
			expect(hasModelInCache('/models/model1.glb')).toBe(false);
			expect(hasModelInCache('/models/model2.glb')).toBe(false);
			expect(hasModelInCache('/models/model3.glb')).toBe(false);
		});
	});

	describe('Cache statistics', () => {
		it('should report cache size and keys', () => {
			const scene1 = new THREE.Group();
			const scene2 = new THREE.Group();

			cacheModel('/models/model1.glb', scene1);
			cacheModel('/models/model2.glb', scene2);

			const stats = getModelCacheStats();
			expect(stats.size).toBe(2);
			expect(stats.keys).toEqual(['/models/model1.glb', '/models/model2.glb']);
		});

		it('should report keys in LRU order', () => {
			const scene1 = new THREE.Group();
			const scene2 = new THREE.Group();
			const scene3 = new THREE.Group();

			cacheModel('/models/model1.glb', scene1);
			cacheModel('/models/model2.glb', scene2);
			cacheModel('/models/model3.glb', scene3);

			// Access model1 to move it to end
			getCachedModel('/models/model1.glb');

			const stats = getModelCacheStats();
			expect(stats.keys).toEqual([
				'/models/model2.glb',
				'/models/model3.glb',
				'/models/model1.glb'
			]);
		});
	});

	describe('Resource disposal', () => {
		it('should dispose geometry when model is evicted', () => {
			const scene = new THREE.Group();
			const geometry = new THREE.BoxGeometry(1, 1, 1);
			const material = new THREE.MeshBasicMaterial();
			const mesh = new THREE.Mesh(geometry, material);
			scene.add(mesh);

			const disposeSpy = vi.spyOn(geometry, 'dispose');

			cacheModel('/models/test.glb', scene);
			evictModel('/models/test.glb');

			expect(disposeSpy).toHaveBeenCalled();
		});

		it('should dispose material when model is evicted', () => {
			const scene = new THREE.Group();
			const geometry = new THREE.BoxGeometry(1, 1, 1);
			const material = new THREE.MeshBasicMaterial();
			const mesh = new THREE.Mesh(geometry, material);
			scene.add(mesh);

			const disposeSpy = vi.spyOn(material, 'dispose');

			cacheModel('/models/test.glb', scene);
			evictModel('/models/test.glb');

			expect(disposeSpy).toHaveBeenCalled();
		});

		it('should dispose line segments', () => {
			const scene = new THREE.Group();
			const geometry = new THREE.BufferGeometry();
			const material = new THREE.LineBasicMaterial();
			const lines = new THREE.LineSegments(geometry, material);
			scene.add(lines);

			const geometryDisposeSpy = vi.spyOn(geometry, 'dispose');
			const materialDisposeSpy = vi.spyOn(material, 'dispose');

			cacheModel('/models/test.glb', scene);
			evictModel('/models/test.glb');

			expect(geometryDisposeSpy).toHaveBeenCalled();
			expect(materialDisposeSpy).toHaveBeenCalled();
		});

		it('should dispose all resources when clearing cache', () => {
			const scene1 = new THREE.Group();
			const geometry1 = new THREE.BoxGeometry(1, 1, 1);
			const material1 = new THREE.MeshBasicMaterial();
			scene1.add(new THREE.Mesh(geometry1, material1));

			const scene2 = new THREE.Group();
			const geometry2 = new THREE.SphereGeometry(1);
			const material2 = new THREE.MeshStandardMaterial();
			scene2.add(new THREE.Mesh(geometry2, material2));

			const disposeSpy1 = vi.spyOn(geometry1, 'dispose');
			const disposeSpy2 = vi.spyOn(geometry2, 'dispose');

			cacheModel('/models/model1.glb', scene1);
			cacheModel('/models/model2.glb', scene2);

			clearModelCache();

			expect(disposeSpy1).toHaveBeenCalled();
			expect(disposeSpy2).toHaveBeenCalled();
		});
	});

	describe('Complex model structures', () => {
		it('should handle nested groups', () => {
			const rootScene = new THREE.Group();
			const childGroup = new THREE.Group();
			const grandchildGroup = new THREE.Group();

			const geometry = new THREE.BoxGeometry(1, 1, 1);
			const material = new THREE.MeshBasicMaterial();
			grandchildGroup.add(new THREE.Mesh(geometry, material));
			childGroup.add(grandchildGroup);
			rootScene.add(childGroup);

			const disposeSpy = vi.spyOn(geometry, 'dispose');

			cacheModel('/models/nested.glb', rootScene);
			evictModel('/models/nested.glb');

			// Should traverse and dispose nested geometry
			expect(disposeSpy).toHaveBeenCalled();
		});

		it('should handle multiple materials on a mesh', () => {
			const scene = new THREE.Group();
			const geometry = new THREE.BoxGeometry(1, 1, 1);
			const materials = [
				new THREE.MeshBasicMaterial(),
				new THREE.MeshStandardMaterial()
			];
			const mesh = new THREE.Mesh(geometry, materials);
			scene.add(mesh);

			const disposeSpy1 = vi.spyOn(materials[0], 'dispose');
			const disposeSpy2 = vi.spyOn(materials[1], 'dispose');

			cacheModel('/models/test.glb', scene);
			evictModel('/models/test.glb');

			expect(disposeSpy1).toHaveBeenCalled();
			expect(disposeSpy2).toHaveBeenCalled();
		});
	});
});

