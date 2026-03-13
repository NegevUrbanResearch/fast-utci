import { describe, it, expect } from 'vitest';

describe('WebGPU Renderer', () => {
	it('should import WebGPURenderer from three/webgpu', async () => {
		// Verify the import path works after Three.js upgrade
		const THREE = await import('three/webgpu');
		expect(THREE.WebGPURenderer).toBeDefined();
	});
});
