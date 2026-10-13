import * as THREE from 'three';
import { MeshBasicNodeMaterial } from 'three/webgpu';
import { float, instanceIndex, texture, uint, vec2 } from 'three/tsl';
import { createActiveUtciPositionNode } from './activeComputeBufferUtciSurfaceMesh';
import { createActiveMaskUtciInstancedSurfaceGeometry } from './activeMaskUtciSurfaceGeometry';
import type { ActiveCellsUtciGridLayout } from './utciGridLayoutTopology';
import { assertTslInstanceIndexSupport } from './utciSurfaceRenderStrategy';

const TEXTURE_WIDTH_LIMIT = 2048;
const TEXTURE_HEIGHT_LIMIT = 8192;

function validateActiveColors(layout: ActiveCellsUtciGridLayout, colors: Float32Array): void {
	if (
		!Number.isSafeInteger(layout.numPositions) ||
		layout.numPositions < 1 ||
		layout.numPositions > TEXTURE_WIDTH_LIMIT * TEXTURE_HEIGHT_LIMIT ||
		layout.activeCanonicalIndices.length !== layout.numPositions
	) {
		throw new Error('Active UTCI texture point count exceeds the supported texture shape.');
	}
	if (colors.length !== layout.numPositions * 3) {
		throw new Error('Active UTCI texture colors must contain three components per active point.');
	}
}

function fillActiveColors(target: Uint8Array, colors: Float32Array): void {
	for (let pointIndex = 0; pointIndex < colors.length / 3; pointIndex += 1) {
		const offset = pointIndex * 4;
		const colorOffset = pointIndex * 3;
		target[offset] = Math.floor(colors[colorOffset] * 255);
		target[offset + 1] = Math.floor(colors[colorOffset + 1] * 255);
		target[offset + 2] = Math.floor(colors[colorOffset + 2] * 255);
		target[offset + 3] = 255;
	}
}

export function createActiveDataTextureUtciSurfaceMesh(
	layout: ActiveCellsUtciGridLayout,
	colors: Float32Array
): THREE.Mesh {
	validateActiveColors(layout, colors);
	assertTslInstanceIndexSupport();
	const width = Math.min(layout.numPositions, TEXTURE_WIDTH_LIMIT);
	const height = Math.ceil(layout.numPositions / width);
	const pixels = new Uint8Array(width * height * 4);
	fillActiveColors(pixels, colors);
	const colorTexture = new THREE.DataTexture(pixels, width, height, THREE.RGBAFormat);
	colorTexture.needsUpdate = true;
	colorTexture.flipY = false;
	colorTexture.generateMipmaps = false;
	colorTexture.magFilter = THREE.NearestFilter;
	colorTexture.minFilter = THREE.NearestFilter;
	colorTexture.colorSpace = THREE.SRGBColorSpace;
	const column = instanceIndex.mod(uint(width));
	const row = instanceIndex.div(uint(width));
	const uv = vec2(float(column).add(0.5).div(width), float(row).add(0.5).div(height));
	const material = new MeshBasicNodeMaterial({
		map: colorTexture,
		transparent: true,
		opacity: 0.9,
		side: THREE.DoubleSide,
		depthTest: true,
		depthWrite: false,
		toneMapped: false
	});
	material.colorNode = texture(colorTexture, uv).rgb;
	material.positionNode = createActiveUtciPositionNode(layout);
	const mesh = new THREE.Mesh(createActiveMaskUtciInstancedSurfaceGeometry(layout), material);
	mesh.name = 'UTCI Active Texture Overlay';
	mesh.raycast = () => {};
	mesh.userData.raycastDisabledReason = 'active-instanced UTCI surface uses sparse tooltip lookup';
	layout.texture = colorTexture;
	return mesh;
}

export function updateActiveDataTextureUtciSurfaceMesh(
	mesh: THREE.Mesh,
	layout: ActiveCellsUtciGridLayout,
	colors: Float32Array
): boolean {
	validateActiveColors(layout, colors);
	const previous = mesh.userData.utciLayout as ActiveCellsUtciGridLayout | undefined;
	if (
		previous?.renderTopology !== 'active-cells' ||
		previous.activeMaskSignature !== layout.activeMaskSignature ||
		previous.numPositions !== layout.numPositions ||
		previous.coordinateSystem !== layout.coordinateSystem ||
		previous.width !== layout.width ||
		previous.height !== layout.height ||
		previous.gridSize !== layout.gridSize ||
		!previous.texture
	) {
		return false;
	}
	fillActiveColors(previous.texture.image.data as Uint8Array, colors);
	previous.texture.needsUpdate = true;
	layout.texture = previous.texture;
	return true;
}
