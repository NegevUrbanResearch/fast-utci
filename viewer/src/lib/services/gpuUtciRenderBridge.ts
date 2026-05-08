import * as THREE from 'three';
import { MeshBasicNodeMaterial, StorageBufferAttribute } from 'three/webgpu';
import { storage, vertexIndex } from 'three/tsl';
import type { UtciGridLayout } from './pointCloudService';

export type SyntheticGpuUtciBridge = {
	group: THREE.Group;
	dispose: () => void;
};

type SyntheticGpuUtciBridgeOptions = {
	center: THREE.Vector3;
	size: THREE.Vector3;
};

export interface GpuNativeUtciSurfaceMeshOptions {
	layout: UtciGridLayout;
	colors: Float32Array;
	opacity?: number;
}

interface GpuNativeUtciSurfaceState {
	colorStorageAttribute: StorageBufferAttribute;
	width: number;
	height: number;
	gridSize: number;
	vertexCount: number;
	source: 'cpu-uploaded-selected-hour' | 'compute-buffer-selected-hour';
}

const MIN_SPAN = 12;
const GRID_SEGMENTS = 14;
const PROTOTYPE_ELEVATION = 1.5;
const DEFAULT_SURFACE_OPACITY = 0.9;
const SURFACE_VERTICES_PER_CELL = 6;
const SURFACE_COLOR_COMPONENTS = 4;
const GPU_NATIVE_SURFACE_STATE_KEY = 'gpuNativeUtciSurfaceState';
const workingColor = new THREE.Color();

function createStorageBackedColorArray(geometry: THREE.BufferGeometry, span: number): Float32Array {
	const positionAttribute = geometry.getAttribute('position');
	if (!positionAttribute) {
		throw new Error('Synthetic bridge geometry is missing positions.');
	}

	const colors = new Float32Array(positionAttribute.count * 4);
	for (let index = 0; index < positionAttribute.count; index += 1) {
		const x = positionAttribute.getX(index) / span;
		const y = positionAttribute.getY(index) / span;
		const offset = index * 4;

		colors[offset] = 0.25 + 0.75 * (x + 0.5);
		colors[offset + 1] = 0.2 + 0.8 * (y + 0.5);
		colors[offset + 2] = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin((x - y) * Math.PI * 3));
		colors[offset + 3] = 0.92;
	}

	return colors;
}

export function createSyntheticGpuUtciBridge(
	options: SyntheticGpuUtciBridgeOptions
): SyntheticGpuUtciBridge {
	const span = Math.max(options.size.x, options.size.z, MIN_SPAN);
	const geometry = new THREE.PlaneGeometry(span, span, GRID_SEGMENTS, GRID_SEGMENTS).toNonIndexed();
	const colorArray = createStorageBackedColorArray(geometry, span);
	const colorStorageAttribute = new StorageBufferAttribute(colorArray, 4);
	const colorStorage = storage(colorStorageAttribute, 'vec4', colorStorageAttribute.count).toReadOnly();

	const material = new MeshBasicNodeMaterial({
		side: THREE.DoubleSide,
		transparent: true
	});
	material.colorNode = colorStorage.element(vertexIndex).xyz;
	material.opacityNode = colorStorage.element(vertexIndex).w;

	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'SyntheticGpuUtciBridgeMesh';
	mesh.frustumCulled = false;
	mesh.rotation.x = -Math.PI / 2;
	mesh.position.copy(options.center);
	mesh.position.y += Math.max(options.size.y * 0.05, PROTOTYPE_ELEVATION);

	const group = new THREE.Group();
	group.name = 'SyntheticGpuUtciBridge';
	group.add(mesh);

	return {
		group,
		dispose: () => {
			group.removeFromParent();
			geometry.dispose();
			material.dispose();
		}
	};
}

export function createGpuNativeUtciSurfaceMesh(
	options: GpuNativeUtciSurfaceMeshOptions
): THREE.Mesh {
	const geometry = createGpuNativeSurfaceGeometry(options.layout);
	const vertexCount = geometry.getAttribute('position').count;
	const colorArray = new Float32Array(vertexCount * SURFACE_COLOR_COMPONENTS);
	const opacity = options.opacity ?? DEFAULT_SURFACE_OPACITY;

	fillGpuNativeSurfaceVertexColors(colorArray, options.layout, options.colors, opacity);

	const colorStorageAttribute = new StorageBufferAttribute(colorArray, SURFACE_COLOR_COMPONENTS);
	const colorStorage = storage(
		colorStorageAttribute,
		'vec4',
		colorStorageAttribute.count
	).toReadOnly();

	const material = new MeshBasicNodeMaterial({
		side: THREE.DoubleSide,
		transparent: true,
		depthTest: true,
		depthWrite: false
	});
	material.colorNode = colorStorage.element(vertexIndex).xyz;
	material.opacityNode = colorStorage.element(vertexIndex).w;
	material.toneMapped = false;

	const mesh = new THREE.Mesh(geometry, material);
	mesh.name = 'GpuNativeUtciSurfaceMesh';
	mesh.frustumCulled = false;
	mesh.renderOrder = 2;
	mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] = {
		colorStorageAttribute,
		width: options.layout.width,
		height: options.layout.height,
		gridSize: options.layout.gridSize,
		vertexCount,
		source: 'cpu-uploaded-selected-hour'
	} satisfies GpuNativeUtciSurfaceState;

	return mesh;
}

export function updateGpuNativeUtciSurfaceMesh(
	mesh: THREE.Mesh,
	options: GpuNativeUtciSurfaceMeshOptions
): boolean {
	const state = mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as GpuNativeUtciSurfaceState | undefined;
	if (!state) {
		console.warn('[UTCI] Missing gpuNative surface state. Recreate the mesh.');
		return false;
	}

	const expectedVertexCount = options.layout.width * options.layout.height * SURFACE_VERTICES_PER_CELL;
	if (
		state.width !== options.layout.width ||
		state.height !== options.layout.height ||
		state.gridSize !== options.layout.gridSize ||
		state.vertexCount !== expectedVertexCount
	) {
		console.warn('[UTCI] gpuNative surface layout changed; recreate the mesh.');
		return false;
	}

	const colorArray = state.colorStorageAttribute.array as Float32Array;
	fillGpuNativeSurfaceVertexColors(
		colorArray,
		options.layout,
		options.colors,
		options.opacity ?? DEFAULT_SURFACE_OPACITY
	);
	state.colorStorageAttribute.needsUpdate = true;
	return true;
}

export function disposeGpuNativeUtciSurfaceMesh(mesh: THREE.Mesh): void {
	mesh.removeFromParent();

	const material = mesh.material;
	if (Array.isArray(material)) {
		for (const entry of material) {
			entry.dispose();
		}
	} else {
		material.dispose();
	}

	mesh.geometry.dispose();
	delete mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY];
}

export function getGpuNativeUtciSurfaceSource(mesh: THREE.Mesh): string | undefined {
	return (mesh.userData[GPU_NATIVE_SURFACE_STATE_KEY] as GpuNativeUtciSurfaceState | undefined)
		?.source;
}

function createGpuNativeSurfaceGeometry(layout: UtciGridLayout): THREE.BufferGeometry {
	const geometry = new THREE.BufferGeometry();
	const planeWidth = layout.width * layout.gridSize;
	const planeHeight = layout.height * layout.gridSize;
	const halfWidth = planeWidth / 2;
	const halfHeight = planeHeight / 2;
	const positions = new Float32Array(
		layout.width * layout.height * SURFACE_VERTICES_PER_CELL * 3
	);

	let offset = 0;
	for (let row = 0; row < layout.height; row += 1) {
		const z0 = -halfHeight + row * layout.gridSize;
		const z1 = z0 + layout.gridSize;

		for (let col = 0; col < layout.width; col += 1) {
			const x0 = -halfWidth + col * layout.gridSize;
			const x1 = x0 + layout.gridSize;

			positions[offset++] = x0;
			positions[offset++] = 0;
			positions[offset++] = z1;
			positions[offset++] = x1;
			positions[offset++] = 0;
			positions[offset++] = z1;
			positions[offset++] = x0;
			positions[offset++] = 0;
			positions[offset++] = z0;

			positions[offset++] = x1;
			positions[offset++] = 0;
			positions[offset++] = z1;
			positions[offset++] = x1;
			positions[offset++] = 0;
			positions[offset++] = z0;
			positions[offset++] = x0;
			positions[offset++] = 0;
			positions[offset++] = z0;
		}
	}

	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geometry.computeBoundingBox();
	geometry.computeBoundingSphere();
	return geometry;
}

function fillGpuNativeSurfaceVertexColors(
	target: Float32Array,
	layout: UtciGridLayout,
	colors: Float32Array,
	opacity: number
): void {
	target.fill(0);

	const cellColors = new Float32Array(layout.width * layout.height * SURFACE_COLOR_COMPONENTS);
	for (let index = 0; index < layout.numPositions; index += 1) {
		const cellIndex = layout.indexToRow[index] * layout.width + layout.indexToColumn[index];
		const colorOffset = cellIndex * SURFACE_COLOR_COMPONENTS;
		const sourceOffset = index * 3;
		const linearColor = workingColor.setRGB(
			colors[sourceOffset],
			colors[sourceOffset + 1],
			colors[sourceOffset + 2],
			THREE.SRGBColorSpace
		);

		cellColors[colorOffset] = linearColor.r;
		cellColors[colorOffset + 1] = linearColor.g;
		cellColors[colorOffset + 2] = linearColor.b;
		cellColors[colorOffset + 3] = opacity;
	}

	let targetOffset = 0;
	for (let row = 0; row < layout.height; row += 1) {
		for (let col = 0; col < layout.width; col += 1) {
			const cellOffset = (row * layout.width + col) * SURFACE_COLOR_COMPONENTS;
			const r = cellColors[cellOffset];
			const g = cellColors[cellOffset + 1];
			const b = cellColors[cellOffset + 2];
			const a = cellColors[cellOffset + 3];

			for (let vertex = 0; vertex < SURFACE_VERTICES_PER_CELL; vertex += 1) {
				target[targetOffset++] = r;
				target[targetOffset++] = g;
				target[targetOffset++] = b;
				target[targetOffset++] = a;
			}
		}
	}
}
