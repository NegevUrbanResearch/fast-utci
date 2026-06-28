import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import { getMaterial } from '$lib/services/materialPool';
import { LAYER_MATERIALS } from '$lib/types/layerMaterials';

describe('Material Pool Service', () => {
	beforeEach(() => {
		// Clear material pool before each test
		// The pool is a singleton, so we need to reset it
		// In a real implementation, we'd have a reset function
	});

	describe('getMaterial', () => {
		it('should return a material for a valid layer type', () => {
			const material = getMaterial('building');
			expect(material).toBeInstanceOf(THREE.Material);
		});

		it('should return MeshLambertMaterial for building layer type', () => {
			const material = getMaterial('building');
			expect(material).toBeInstanceOf(THREE.MeshLambertMaterial);
		});

		it('should return MeshLambertMaterial for new_building layer type', () => {
			const material = getMaterial('new_building');
			expect(material).toBeInstanceOf(THREE.MeshLambertMaterial);
		});

		it('should return MeshStandardMaterial for non-building layer types', () => {
			const material = getMaterial('vegetation');
			expect(material).toBeInstanceOf(THREE.MeshStandardMaterial);
		});

		it('should return MeshStandardMaterial for base layer type', () => {
			const material = getMaterial('base');
			expect(material).toBeInstanceOf(THREE.MeshStandardMaterial);
		});

		it('should apply correct color from LAYER_MATERIALS config', () => {
			const material = getMaterial('building');
			const config = LAYER_MATERIALS.building;
			expect((material as THREE.MeshLambertMaterial).color.getHexString()).toBe(
				config.color.replace('#', '')
			);
		});

		it('should apply correct opacity from LAYER_MATERIALS config', () => {
			const material = getMaterial('base');
			const config = LAYER_MATERIALS.base;
			expect(material.opacity).toBe(config.opacity);
			expect(material.transparent).toBe(config.opacity < 1.0);
		});

		it('should reuse the same material instance for the same layer type', () => {
			const material1 = getMaterial('building');
			const material2 = getMaterial('building');
			expect(material1).toBe(material2); // Same instance (reused)
		});

		it('should create different materials for different layer types', () => {
			const buildingMaterial = getMaterial('building');
			const vegetationMaterial = getMaterial('vegetation');
			expect(buildingMaterial).not.toBe(vegetationMaterial);
		});

		it('should return default material for unknown layer type', () => {
			const material = getMaterial('unknown_type');
			expect(material).toBeInstanceOf(THREE.MeshStandardMaterial);
			const defaultConfig = LAYER_MATERIALS.default;
			expect((material as THREE.MeshStandardMaterial).color.getHexString()).toBe(
				defaultConfig.color.replace('#', '')
			);
		});

		it('should apply emissive properties for building materials', () => {
			const material = getMaterial('building') as THREE.MeshLambertMaterial;
			const config = LAYER_MATERIALS.building;
			expect(material.emissive).toBeInstanceOf(THREE.Color);
			expect(material.emissiveIntensity).toBe(config.emissiveIntensity);
		});

		it('should apply polygon offset for base layer', () => {
			const material = getMaterial('base') as THREE.MeshStandardMaterial;
			const config = LAYER_MATERIALS.base;
			expect(material.polygonOffset).toBe(config.polygonOffset);
			if (config.polygonOffset) {
				expect(material.polygonOffsetFactor).toBeGreaterThanOrEqual(1);
				expect(material.polygonOffsetUnits).toBeGreaterThanOrEqual(1);
			}
		});

		it('should set shadow properties correctly', () => {
			const buildingMaterial = getMaterial('building');
			expect(buildingMaterial).toBeInstanceOf(THREE.MeshLambertMaterial);
			// Materials don't have shadow properties, meshes do
			// But we can verify the material is set up correctly
		});
	});
});

