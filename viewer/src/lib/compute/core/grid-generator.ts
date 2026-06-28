import * as THREE from 'three';

export interface RectangularBounds2D {
  min: [number, number];
  max: [number, number];
}

export interface RectangularGridResult {
  points: THREE.Vector3[];
  normals: THREE.Vector3[];
}

/**
 * Create an axis-aligned rectangular grid in the XZ plane of the Three.js
 * Y-up world frame, mirroring the behaviour of the Python
 * `create_rectangular_grid` helper as closely as practical.
 *
 * - X corresponds to East/right, Z to North/forward, Y is Up.
 * - Points are placed on a regular lattice within [min, max], inclusive,
 *   using a step of `gridSize` along both X and Z.
 * - All normals point strictly up (0, 1, 0) and are suitable for
 *   pedestrian-level MRT/UTCI sampling.
 */
export function createRectangularGridFromBounds(
  bounds: RectangularBounds2D,
  gridSize: number,
  zHeight: number
): RectangularGridResult {
  const [minX, minZ] = bounds.min;
  const [maxX, maxZ] = bounds.max;

  if (gridSize <= 0) {
    throw new Error('gridSize must be positive');
  }

  const points: THREE.Vector3[] = [];
  const normals: THREE.Vector3[] = [];

  // Match numpy.arange semantics from the Python implementation by ensuring
  // we include the upper bound when it lands exactly on the grid.
  const epsilon = 1e-9;

  for (let x = minX; x <= maxX + epsilon; x += gridSize) {
    for (let z = minZ; z <= maxZ + epsilon; z += gridSize) {
      points.push(new THREE.Vector3(x, zHeight, z));
      normals.push(new THREE.Vector3(0, 1, 0));
    }
  }

  return { points, normals };
}
