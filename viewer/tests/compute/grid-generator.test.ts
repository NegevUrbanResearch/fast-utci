import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { generateGridFromMesh, createRectangularGridFromBounds } from '$lib/compute/grid-generator';

describe('Grid Generator', () => {
  it('should generate points at correct zHeight above horizontal surfaces', () => {
    // Create a 10x10 plane at z=0 (in Three.js horizontal is usually y=0, but fast-utci uses z=up according to Python code and Ladybug)
    // Actually the app uses standard Three.js where Y is up. Let's make a horizontal plane (XZ).
    const geometry = new THREE.PlaneGeometry(10, 10);
    // PlaneGeometry defaults to XY plane. Rotate to XZ plane.
    geometry.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(geometry);

    // Create a grid with 2m resolution and 1.5m zHeight
    // For a 10x10 area and 2m resolution, we expect roughly 5x5 = 25 points,
    // or maybe 6x6 = 36 if it includes edges.
    const result = generateGridFromMesh(mesh, 2, 1.5);
    
    expect(result.points.length).toBeGreaterThan(0);
    expect(result.normals.length).toBe(result.points.length);

    // Check that points are ~1.5m above the surface (y = 1.5 since Y is up)
    for (const pt of result.points) {
      expect(pt.y).toBeCloseTo(1.5, 3);
    }
  });

  it('should generate upwards facing normals', () => {
    const geometry = new THREE.PlaneGeometry(10, 10);
    geometry.rotateX(-Math.PI / 2);
    const mesh = new THREE.Mesh(geometry);

    const result = generateGridFromMesh(mesh, 2, 1);
    
    expect(result.normals.length).toBeGreaterThan(0);
    for (const normal of result.normals) {
      expect(normal.x).toBeCloseTo(0, 3);
      expect(normal.y).toBeCloseTo(1, 3); // Upwards
      expect(normal.z).toBeCloseTo(0, 3);
    }
  });

  it('should ignore vertical walls based on normal angle', () => {
    // Box 10x10x10 on floor. Only top face should be used.
    const geometry = new THREE.BoxGeometry(10, 10, 10);
    // Shift box so the bottom is at y=0, top is at y=10
    geometry.translate(0, 5, 0);
    const mesh = new THREE.Mesh(geometry);

    const result = generateGridFromMesh(mesh, 2, 1.5);
    
    // Test that all generated points are at y = 10 + 1.5 = 11.5
    expect(result.points.length).toBeGreaterThan(0);
    for (const pt of result.points) {
      expect(pt.y).toBeCloseTo(11.5, 3);
    }
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
