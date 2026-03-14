import { describe, it, expect } from 'vitest';
import { analysisPositionsToWorld } from '$lib/parity/analysisToWorld';

describe('analysisPositionsToWorld', () => {
	it('converts xy_ground (x,y,z) to world (x, z, -y)', () => {
		const analysis = new Float32Array([10, 20, 1.5]); // one point
		const world = analysisPositionsToWorld(analysis, 'xy_ground');
		expect(world[0]).toBe(10);
		expect(world[1]).toBe(1.5);
		expect(world[2]).toBe(-20);
	});

	it('converts multiple points', () => {
		const analysis = new Float32Array([0, 0, 0, 1, 1, 1]);
		const world = analysisPositionsToWorld(analysis, 'xy_ground');
		expect(world.length).toBe(6);
		expect(world[0]).toBe(0);
		expect(world[1]).toBe(0);
		expect(world[2] === 0).toBe(true); // -0 for (0,0,0) → (0,0,-0)
		expect(world[3]).toBe(1);
		expect(world[4]).toBe(1);
		expect(world[5]).toBe(-1);
	});
});
