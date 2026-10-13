import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { createActiveDataTextureUtciSurfaceMesh } from '$lib/services/activeDataTextureUtciSurfaceMesh';
import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';

function layoutForCount(numPositions: number): ActiveCellsUtciGridLayout {
	return {
		renderTopology: 'active-cells',
		width: 4096,
		height: 2,
		gridSize: 1,
		coordinateSystem: 'xy_ground',
		numPositions,
		minX: 0,
		minZ: 0,
		minY: 0,
		maxY: 0,
		centerX: 0,
		centerZ: 0,
		baseY: 0,
		renderCellCount: numPositions,
		canonicalCellCount: 8192,
		activeMaskSignature: 'sparse-test',
		activeCanonicalIndices: Uint32Array.from(
			{ length: numPositions },
			(_, index) => index * 2
		)
	};
}

describe('active DataTexture surface', () => {
	it('packs active colors into bounded rows and leaves padding outside the instance count', () => {
		const layout = layoutForCount(2049);
		const colors = new Float32Array(2049 * 3);
		colors.set([0.5, 0.25, 1]);
		colors.set([0.2, 0.4, 0.6], 2048 * 3);
		const mesh = createActiveDataTextureUtciSurfaceMesh(layout, colors);
		const texture = layout.texture!;
		const pixels = texture.image.data as Uint8Array;
		expect(texture.image.width).toBe(2048);
		expect(texture.image.height).toBe(2);
		expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(2049);
		expect(Array.from(pixels.slice(0, 4))).toEqual([127, 63, 255, 255]);
		expect(Array.from(pixels.slice(2048 * 4, 2049 * 4))).toEqual([51, 102, 153, 255]);
		expect(Array.from(pixels.slice(2049 * 4))).toEqual(new Array(2047 * 4).fill(0));
		expect(texture.minFilter).toBe(THREE.NearestFilter);
		expect(texture.magFilter).toBe(THREE.NearestFilter);
		expect(mesh.geometry.getAttribute('activeCanonicalIndex').array).toBe(layout.activeCanonicalIndices);
		texture.dispose();
		(mesh.material as THREE.Material).dispose();
		mesh.geometry.dispose();
	});

	it('rejects malformed color arrays and impossible counts before allocating assets', () => {
		expect(() => createActiveDataTextureUtciSurfaceMesh(layoutForCount(2), new Float32Array(3)))
			.toThrow('three components per active point');
		const layout = layoutForCount(1);
		layout.numPositions = 2048 * 8192 + 1;
		expect(() => createActiveDataTextureUtciSurfaceMesh(layout, new Float32Array()))
			.toThrow('supported texture shape');
		expect(() => createActiveDataTextureUtciSurfaceMesh(layoutForCount(0), new Float32Array()))
			.toThrow('supported texture shape');
	});
});
