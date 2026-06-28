import * as THREE from 'three';
import type { ActiveCellsUtciGridLayout } from '$lib/services/utciGridLayoutTopology';
import { getActiveMaskGridCell } from '$lib/services/utciGridLayoutTopology';

export type ActiveMaskUtciSurfaceShape = {
	label: string;
	activePointCount: number;
	canonicalCellCount: number;
	canonicalWidth: number;
	canonicalHeight: number;
};

export const INNOVATION_DISTRICT_05M_ACTIVE_MASK_SURFACE_SHAPE = {
	label: 'innovation-district-0.5m-active-mask',
	activePointCount: 10_218_798,
	canonicalCellCount: 23_871_025,
	canonicalWidth: 5_195,
	canonicalHeight: 4_595
} as const satisfies ActiveMaskUtciSurfaceShape;

export type ActiveMaskUtciInstancedGeometryByteEstimate = {
	vertexBufferBytes: number;
	indexBufferBytes: number;
	activeCanonicalIndexAttributeBytes: number;
	totalBytes: number;
};

export type ActiveMaskUtciCanonicalCellCenter = {
	x: number;
	z: number;
};

const QUAD_VERTEX_COMPONENTS = 3;
const SHARED_QUAD_VERTEX_COUNT = 4;
const SHARED_QUAD_INDEX_COUNT = 6;
const FLOAT32_BYTES = 4;
const UINT32_BYTES = 4;

function setActiveMaskUtciInstancedSurfaceAnalyticBounds(
	geometry: THREE.BufferGeometry,
	layout: ActiveCellsUtciGridLayout
): void {
	const halfWidth = (layout.width * layout.gridSize) / 2;
	const halfHeight = (layout.height * layout.gridSize) / 2;
	geometry.boundingBox = new THREE.Box3(
		new THREE.Vector3(-halfWidth, 0, -halfHeight),
		new THREE.Vector3(halfWidth, 0, halfHeight)
	);
	geometry.boundingSphere = new THREE.Sphere(
		new THREE.Vector3(0, 0, 0),
		Math.hypot(halfWidth, halfHeight)
	);
}

export function getActiveMaskUtciCanonicalCellCenter(params: {
	layout: ActiveCellsUtciGridLayout;
	canonicalIndex: number;
}): ActiveMaskUtciCanonicalCellCenter {
	const { layout, canonicalIndex } = params;
	const { row, col } = getActiveMaskGridCell({
		canonicalIndex,
		width: layout.width,
		height: layout.height,
		coordinateSystem: layout.coordinateSystem
	});
	const halfWidth = (layout.width * layout.gridSize) / 2;
	const halfHeight = (layout.height * layout.gridSize) / 2;

	return {
		x: -halfWidth + (col + 0.5) * layout.gridSize,
		z: -halfHeight + (row + 0.5) * layout.gridSize
	};
}

export function estimateActiveMaskUtciInstancedGeometryBytes(
	layout: Pick<ActiveCellsUtciGridLayout, 'activeCanonicalIndices'>
): ActiveMaskUtciInstancedGeometryByteEstimate {
	const vertexBufferBytes =
		SHARED_QUAD_VERTEX_COUNT * QUAD_VERTEX_COMPONENTS * FLOAT32_BYTES;
	const indexBufferBytes = SHARED_QUAD_INDEX_COUNT * UINT32_BYTES;
	const activeCanonicalIndexAttributeBytes = layout.activeCanonicalIndices.byteLength;
	return {
		vertexBufferBytes,
		indexBufferBytes,
		activeCanonicalIndexAttributeBytes,
		totalBytes: vertexBufferBytes + indexBufferBytes + activeCanonicalIndexAttributeBytes
	};
}

export function createActiveMaskUtciInstancedSurfaceGeometry(
	layout: ActiveCellsUtciGridLayout
): THREE.InstancedBufferGeometry {
	const halfCell = layout.gridSize / 2;
	const geometry = new THREE.InstancedBufferGeometry();
	const positions = new Float32Array([
		-halfCell,
		0,
		-halfCell,
		halfCell,
		0,
		-halfCell,
		-halfCell,
		0,
		halfCell,
		halfCell,
		0,
		halfCell
	]);
	const indices = new Uint32Array([2, 3, 0, 3, 1, 0]);

	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geometry.setIndex(new THREE.BufferAttribute(indices, 1));
	geometry.setAttribute(
		'activeCanonicalIndex',
		new THREE.InstancedBufferAttribute(layout.activeCanonicalIndices, 1)
	);
	geometry.instanceCount = layout.activeCanonicalIndices.length;
	setActiveMaskUtciInstancedSurfaceAnalyticBounds(geometry, layout);
	return geometry;
}

export function disposeActiveMaskUtciInstancedSurfaceGeometry(
	geometry: THREE.BufferGeometry | null | undefined
): void {
	geometry?.dispose();
}
