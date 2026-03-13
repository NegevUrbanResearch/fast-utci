import * as THREE from 'three';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';

// Add BVH extension to THREE.BufferGeometry
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export interface GridResult {
  points: THREE.Vector3[];
  normals: THREE.Vector3[];
}

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

/**
 * Generates a grid of sensor points strictly above horizontal (or near-horizontal) surfaces.
 * Evaluated point locations are offset by zHeight along the surface normal (usually UP).
 * 
 * @param mesh The Three.js mesh to generate the grid over.
 * @param resolution The grid cell size in meters.
 * @param zHeight The height offset for the sensor points (e.g., 1.5m for standing head height).
 * @param maxSlopeDegrees The maximum slope angle (from horizontal) to consider a surface "walkable". Default is 45.
 * @returns Object containing arrays of Vector3 points and normals.
 */
export function generateGridFromMesh(
  mesh: THREE.Mesh,
  resolution: number,
  zHeight: number,
  maxSlopeDegrees: number = 45
): GridResult {
  // Ensure the geometry has a bounding box
  mesh.geometry.computeBoundingBox();
  const bbox = mesh.geometry.boundingBox;
  if (!bbox) return { points: [], normals: [] };

  // Generate BVH if not present for faster raycasting
  if (!mesh.geometry.boundsTree) {
    mesh.geometry.computeBoundsTree();
  }

  // Convert max slope to radians for dot product comparison with UP vector
  const maxSlopeRad = THREE.MathUtils.degToRad(maxSlopeDegrees);
  const minNormalY = Math.cos(maxSlopeRad); // Dot product of UP (0,1,0) and allowable normal

  const points: THREE.Vector3[] = [];
  const normals: THREE.Vector3[] = [];

  // Determine grid boundaries based on the bounding box transformed into
  // Three.js world space. In this convention Y is Up, X is East/right, and
  // Z is North/forward; all grid points and normals are prepared in this
  // Y-up frame so they are directly compatible with the BVH and WebGPU.
  const worldBBox = bbox.clone().applyMatrix4(mesh.matrixWorld);

  const startX = Math.floor(worldBBox.min.x / resolution) * resolution;
  const endX = Math.ceil(worldBBox.max.x / resolution) * resolution;
  const startZ = Math.floor(worldBBox.min.z / resolution) * resolution;
  const endZ = Math.ceil(worldBBox.max.z / resolution) * resolution;

  const raycaster = new THREE.Raycaster();
  raycaster.firstHitOnly = true; // Optimization provided by three-mesh-bvh

  // Ray direction is downwards
  const rayDir = new THREE.Vector3(0, -1, 0);

  // Shoot rays from above the bounding box top
  const originY = worldBBox.max.y + 10; 

  for (let x = startX; x <= endX; x += resolution) {
    for (let z = startZ; z <= endZ; z += resolution) {
      const origin = new THREE.Vector3(x, originY, z);
      raycaster.set(origin, rayDir);

      const intersects = raycaster.intersectObject(mesh);

      if (intersects.length > 0) {
        const hit = intersects[0];
        
        // Check if the surface normal is sufficiently "upward" facing
        if (hit.face && hit.face.normal) {
          // Normal needs to be transformed to world space if mesh is rotated
          const worldNormal = hit.face.normal.clone().transformDirection(mesh.matrixWorld).normalize();

          // Check against minimum Y component (dot product with 0,1,0)
          if (worldNormal.y >= minNormalY) {
            // Valid horizontal surface -> offset point by zHeight
            const sensorPos = hit.point.clone().add(worldNormal.multiplyScalar(zHeight));
            points.push(sensorPos);
            // Default normal for sensors is usually strictly UP (0,1,0) or surface normal.
            // Following Ladybug/Grasshopper typical behaviour, sensor normal is surface normal
            // so directional radiation matches surface orientation.
            normals.push(new THREE.Vector3(0, 1, 0)); // Using standard UP for human body analyses
          }
        }
      }
    }
  }

  return { points, normals };
}
