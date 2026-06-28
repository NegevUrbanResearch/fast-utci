import { describe, it, expect } from 'vitest';
import * as grid from '$lib/compute/core/grid-generator';
import { createRectangularGridFromBounds } from '$lib/compute/core/grid-generator';

describe('Grid Generator', () => {
  it('does not export generateGridFromMesh', () => {
    expect((grid as Record<string, unknown>).generateGridFromMesh).toBeUndefined();
  });

  it('should generate a rectangular grid within bounds with expected count and normals', () => {
    const grid = createRectangularGridFromBounds(
      { min: [0, 0], max: [10, 10] },
      5.0,
      1.1
    );

    // 0,5,10 → 3 steps in each dimension → 3×3 = 9 points
    expect(grid.points.length).toBe(9);
    expect(grid.normals.length).toBe(9);

    grid.points.forEach((p) => {
      expect(p.y).toBeCloseTo(1.1, 6);
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x).toBeLessThanOrEqual(10.000001);
      expect(p.z).toBeGreaterThanOrEqual(0);
      expect(p.z).toBeLessThanOrEqual(10.000001);
    });

    grid.normals.forEach((n) => {
      expect(n.x).toBe(0);
      expect(n.y).toBe(1);
      expect(n.z).toBe(0);
    });
  });
});
