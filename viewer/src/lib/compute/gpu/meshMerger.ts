import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import {
	countTrianglesInGroup,
	MAX_TRIANGLES_FOR_MAIN_THREAD
} from '$lib/compute/gpu/mergeAndBvhWorkerClient';

/**
 * Traverse a Three.js scene group and merge all Mesh children into a
 * single Mesh with combined world-space geometry.
 *
 * Returns null if no meshes are found in the group.
 */
export function mergeSceneMeshes(group: THREE.Group | THREE.Object3D): THREE.Mesh | null {
	const { totalTriangles } = countTrianglesInGroup(group);
	if (totalTriangles > MAX_TRIANGLES_FOR_MAIN_THREAD) {
		console.warn(
			`[MESH MERGER] Refusing main-thread merge: ${(totalTriangles / 1e6).toFixed(2)}M triangles > ${(MAX_TRIANGLES_FOR_MAIN_THREAD / 1e6).toFixed(2)}M cap`
		);
		return null;
	}

	group.updateMatrixWorld(true);

	const geometries: THREE.BufferGeometry[] = [];

	group.traverse((child) => {
		if (child instanceof THREE.Mesh && child.geometry) {
			const cloned = child.geometry.clone();
			cloned.applyMatrix4(child.matrixWorld);

			// mergeGeometries requires all geometries to have the same attributes.
			// Keep only position and index; drop normals/uvs/etc. since grid gen
			// and BVH only need position+index.
			const posAttr = cloned.getAttribute('position');
			const idxAttr = cloned.getIndex();
			if (!posAttr) return;

			const stripped = new THREE.BufferGeometry();
			stripped.setAttribute('position', posAttr);
			if (idxAttr) {
				stripped.setIndex(idxAttr);
			} else {
				// Non-indexed geometry: create trivial index
				const count = posAttr.count;
				const indices = new Uint32Array(count);
				for (let i = 0; i < count; i++) indices[i] = i;
				stripped.setIndex(new THREE.BufferAttribute(indices, 1));
			}

			geometries.push(stripped);
		}
	});

	if (geometries.length === 0) return null;

	const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
	if (!merged) return null;

	// The merged geometry is already in world space; the mesh sits at identity.
	const mesh = new THREE.Mesh(merged);
	mesh.matrixWorld.identity();
	return mesh;
}
