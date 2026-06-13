import * as THREE from 'three';
import { LAYER_MATERIALS } from '$lib/types/layerMaterials';

const ROAD_OUTLINE_NAME = 'road_outline';
const UTCI_OVERLAY_RENDER_ORDER = 2;

function getPrimaryMaterial(mesh: THREE.Mesh): THREE.Material | null {
	if (Array.isArray(mesh.material)) {
		return mesh.material[0] ?? null;
	}
	return mesh.material ?? null;
}

function ensureRoadOutline(mesh: THREE.Mesh): void {
	const config = LAYER_MATERIALS.road;
	const existingOutline = mesh.children.find(
		(child): child is THREE.LineSegments =>
			child instanceof THREE.LineSegments && child.name === ROAD_OUTLINE_NAME
	);
	if (existingOutline) {
		existingOutline.renderOrder = config.renderOrder ?? UTCI_OVERLAY_RENDER_ORDER + 1;
		return;
	}

	const edges = new THREE.EdgesGeometry(
		mesh.geometry,
		config.outlineThresholdAngle ?? 15
	);
	const outlineMaterial = new THREE.LineBasicMaterial({
		color: new THREE.Color(config.outlineColor ?? config.color),
		opacity: config.outlineOpacity ?? 1,
		transparent: (config.outlineOpacity ?? 1) < 1
	});
	outlineMaterial.depthTest = config.outlineDepthTest ?? true;
	outlineMaterial.depthWrite = config.outlineDepthWrite ?? false;
	outlineMaterial.toneMapped = config.outlineToneMapped ?? true;

	const outline = new THREE.LineSegments(edges, outlineMaterial);
	outline.name = ROAD_OUTLINE_NAME;
	outline.renderOrder = config.renderOrder ?? UTCI_OVERLAY_RENDER_ORDER + 1;
	mesh.add(outline);
}

function applyRoadRenderPolicy(mesh: THREE.Mesh): void {
	const config = LAYER_MATERIALS.road;
	mesh.renderOrder = config.renderOrder ?? UTCI_OVERLAY_RENDER_ORDER + 1;

	const material = getPrimaryMaterial(mesh);
	if (material && 'opacity' in material) {
		material.opacity = config.opacity;
		material.transparent = true;
		material.depthWrite = false;
	}

	ensureRoadOutline(mesh);
}

function applyTrainTrackRenderPolicy(mesh: THREE.Mesh): void {
	const config = LAYER_MATERIALS.train_track;
	mesh.renderOrder = config.renderOrder ?? UTCI_OVERLAY_RENDER_ORDER + 1;

	const material = getPrimaryMaterial(mesh);
	if (material && 'depthWrite' in material) {
		material.depthWrite = true;
		material.depthTest = true;
	}
}

export function applyLayerRenderPolicy(mesh: THREE.Mesh, layerType: string): void {
	if (layerType === 'road') {
		applyRoadRenderPolicy(mesh);
		return;
	}

	if (layerType === 'train_track') {
		applyTrainTrackRenderPolicy(mesh);
	}
}
