import * as THREE from 'three';
import type { Group } from 'three';
import {
	buildClassifiedStudyAreaMaskFromProjectedTriangles,
	type ClassifiedStudyAreaMaskResult
} from '$lib/compute/core/studyAreaMask';
import {
	type ClassifiedAnalysisActiveMask,
	type AnalysisCoordinateSystem,
	type AnalysisRectangularBounds,
	type ClassifiedProjectedTriangle2D,
	type ProjectedTriangle2D,
	type StudyAreaMask,
	type SurfaceFlags,
	SURFACE_FLAGS
} from '$lib/types/analysis';

export interface ClassifiedActiveMaskBuildParams {
	model: Group;
	bounds: AnalysisRectangularBounds;
	gridResolution: number;
	coordinateSystem: AnalysisCoordinateSystem;
	gridOriginOffset?: { x: number; y: number; z: number };
	requireClassifiedSurface?: boolean;
}

export function normalizeSurfaceLayerName(value: unknown): string {
	return String(value ?? '')
		.trim()
		.toLowerCase()
		.replace(/[-\s]+/g, '_');
}

function isTrainTrackLayer(normalizedValue: string): boolean {
	return normalizedValue === 'train_track' || normalizedValue === 'train_tracks';
}

function surfaceFlagsForLayer(params: { layerType: unknown; layerName: unknown }): SurfaceFlags | null {
	const layerType = normalizeSurfaceLayerName(params.layerType);
	const layerName = normalizeSurfaceLayerName(params.layerName);
	if (isTrainTrackLayer(layerType) || isTrainTrackLayer(layerName)) {
		return null;
	}
	if (
		layerType === 'building' ||
		layerType === 'existing_buildings' ||
		layerType === 'new_buildings' ||
		layerName === 'building' ||
		layerName === 'existing_buildings' ||
		layerName === 'new_buildings'
	) {
		return SURFACE_FLAGS.buildingFootprint;
	}
	if (layerType === 'base' || layerType === 'ground' || layerName === 'base' || layerName === 'ground') {
		return SURFACE_FLAGS.ground;
	}
	if (
		layerType === 'road' ||
		layerType === 'roads' ||
		layerType === 'street' ||
		layerType === 'streets' ||
		layerType === 'sidewalk' ||
		layerType === 'sidewalks' ||
		layerName === 'street' ||
		layerName === 'streets' ||
		layerName === 'road' ||
		layerName === 'roads' ||
		layerName === 'sidewalk' ||
		layerName === 'sidewalks'
	) {
		return SURFACE_FLAGS.streetSurface;
	}
	return null;
}

function hasSurfaceFlag(flags: SurfaceFlags, flag: SurfaceFlags): boolean {
	return (flags & flag) !== 0;
}

function isSampledSurfaceFlags(flags: SurfaceFlags): boolean {
	return hasSurfaceFlag(flags, SURFACE_FLAGS.ground) || hasSurfaceFlag(flags, SURFACE_FLAGS.streetSurface);
}

function isBuildingFootprintFlags(flags: SurfaceFlags): boolean {
	return hasSurfaceFlag(flags, SURFACE_FLAGS.buildingFootprint);
}

function pushProjectedTriangle(params: {
	triangles: ProjectedTriangle2D[];
	positions: THREE.BufferAttribute | THREE.InterleavedBufferAttribute;
	matrixWorld: THREE.Matrix4;
	originOffset?: { x: number; y: number; z: number };
	va: THREE.Vector3;
	vb: THREE.Vector3;
	vc: THREE.Vector3;
	a: number;
	b: number;
	c: number;
}): void {
	const va = params.va.fromBufferAttribute(params.positions, params.a).applyMatrix4(params.matrixWorld);
	const vb = params.vb.fromBufferAttribute(params.positions, params.b).applyMatrix4(params.matrixWorld);
	const vc = params.vc.fromBufferAttribute(params.positions, params.c).applyMatrix4(params.matrixWorld);
	if (
		!Number.isFinite(va.x) ||
		!Number.isFinite(va.z) ||
		!Number.isFinite(vb.x) ||
		!Number.isFinite(vb.z) ||
		!Number.isFinite(vc.x) ||
		!Number.isFinite(vc.z)
	) {
		return;
	}
	const originOffset = params.originOffset ?? { x: 0, y: 0, z: 0 };
	params.triangles.push([
		va.x - originOffset.x,
		va.z - originOffset.z,
		vb.x - originOffset.x,
		vb.z - originOffset.z,
		vc.x - originOffset.x,
		vc.z - originOffset.z
	]);
}

function extractProjectedTriangles(params: {
	model: Group;
	originOffset?: { x: number; y: number; z: number };
	includeFlags: (flags: SurfaceFlags) => boolean;
}): ClassifiedProjectedTriangle2D[] {
	const triangles: ClassifiedProjectedTriangle2D[] = [];

	params.model.updateWorldMatrix?.(true, true);
	params.model.traverse((child) => {
		if (!(child instanceof THREE.Mesh)) {
			return;
		}
		const flags = surfaceFlagsForLayer({
			layerType: child.userData.layerType,
			layerName: child.userData.layerName ?? child.name
		});
		if (!flags || !params.includeFlags(flags)) {
			return;
		}
		const geometry = child.geometry;
		const positions = geometry.getAttribute('position');
		if (!positions) {
			return;
		}
		child.updateWorldMatrix(true, false);
		const meshTriangles: ProjectedTriangle2D[] = [];
		const index = geometry.getIndex();
		const va = new THREE.Vector3();
		const vb = new THREE.Vector3();
		const vc = new THREE.Vector3();
		if (index) {
			for (let offset = 0; offset + 2 < index.count; offset += 3) {
				pushProjectedTriangle({
					triangles: meshTriangles,
					positions,
					matrixWorld: child.matrixWorld,
					originOffset: params.originOffset,
					va,
					vb,
					vc,
					a: index.getX(offset),
					b: index.getX(offset + 1),
					c: index.getX(offset + 2)
				});
			}
		} else {
			for (let offset = 0; offset + 2 < positions.count; offset += 3) {
				pushProjectedTriangle({
					triangles: meshTriangles,
					positions,
					matrixWorld: child.matrixWorld,
					originOffset: params.originOffset,
					va,
					vb,
					vc,
					a: offset,
					b: offset + 1,
					c: offset + 2
				});
			}
		}

		for (const triangle of meshTriangles) {
			triangles.push({ triangle, flags });
		}
	});

	return triangles;
}

function countProjectedMeshTriangles(params: {
	model: Group;
	originOffset?: { x: number; y: number; z: number };
	includeMesh: (mesh: THREE.Mesh) => boolean;
}): number {
	let triangleCount = 0;

	params.model.updateWorldMatrix?.(true, true);
	params.model.traverse((child) => {
		if (!(child instanceof THREE.Mesh) || !params.includeMesh(child)) {
			return;
		}
		const geometry = child.geometry;
		const positions = geometry.getAttribute('position');
		if (!positions) {
			return;
		}
		child.updateWorldMatrix(true, false);
		const meshTriangles: ProjectedTriangle2D[] = [];
		const index = geometry.getIndex();
		const va = new THREE.Vector3();
		const vb = new THREE.Vector3();
		const vc = new THREE.Vector3();
		if (index) {
			for (let offset = 0; offset + 2 < index.count; offset += 3) {
				pushProjectedTriangle({
					triangles: meshTriangles,
					positions,
					matrixWorld: child.matrixWorld,
					originOffset: params.originOffset,
					va,
					vb,
					vc,
					a: index.getX(offset),
					b: index.getX(offset + 1),
					c: index.getX(offset + 2)
				});
			}
		} else {
			for (let offset = 0; offset + 2 < positions.count; offset += 3) {
				pushProjectedTriangle({
					triangles: meshTriangles,
					positions,
					matrixWorld: child.matrixWorld,
					originOffset: params.originOffset,
					va,
					vb,
					vc,
					a: offset,
					b: offset + 1,
					c: offset + 2
				});
			}
		}
		triangleCount += meshTriangles.length;
	});

	return triangleCount;
}

export function extractSampledSurfaceProjectedTriangles(
	model: Group,
	originOffset?: { x: number; y: number; z: number }
): ClassifiedProjectedTriangle2D[] {
	return extractProjectedTriangles({
		model,
		originOffset,
		includeFlags: isSampledSurfaceFlags
	});
}

export function extractBuildingFootprintProjectedTriangles(
	model: Group,
	originOffset?: { x: number; y: number; z: number }
): ClassifiedProjectedTriangle2D[] {
	return extractProjectedTriangles({
		model,
		originOffset,
		includeFlags: isBuildingFootprintFlags
	});
}

export function buildClassifiedActiveMaskSurface(params: ClassifiedActiveMaskBuildParams):
	| ClassifiedStudyAreaMaskResult
	| undefined {
	const sampledSurfaceTriangles = extractSampledSurfaceProjectedTriangles(
		params.model,
		params.gridOriginOffset
	);
	if (sampledSurfaceTriangles.length === 0) {
		const projectedTriangleCount = countProjectedMeshTriangles({
			model: params.model,
			originOffset: params.gridOriginOffset,
			includeMesh: () => true
		});
		if (params.requireClassifiedSurface && projectedTriangleCount > 0) {
			throw new Error(
				'Classified active-mask surface classification found model triangles but no sampled ground/street surface triangles; refusing to fall back to the full rectangular grid.'
			);
		}
		return undefined;
	}

	const buildingFootprintTriangles = extractBuildingFootprintProjectedTriangles(
		params.model,
		params.gridOriginOffset
	);

	const result = buildClassifiedStudyAreaMaskFromProjectedTriangles({
		bounds: params.bounds,
		gridSize: params.gridResolution,
		coordinateSystem: params.coordinateSystem,
		triangles: [...sampledSurfaceTriangles, ...buildingFootprintTriangles]
	});
	return result;
}

function createClassifiedAnalysisActiveMask(params: {
	activeMask: StudyAreaMask;
	surfaceFlagsByActiveCell: Uint8Array;
}): ClassifiedAnalysisActiveMask {
	const inactivePointCount = params.activeMask.canonicalPointCount - params.activeMask.activePointCount;
	return {
		source: 'base+road',
		canonicalPointCount: params.activeMask.canonicalPointCount,
		activePointCount: params.activeMask.activePointCount,
		inactivePointCount,
		activePointRatio:
			params.activeMask.canonicalPointCount > 0
				? params.activeMask.activePointCount / params.activeMask.canonicalPointCount
				: 0,
		activeMaskChecksum: params.activeMask.maskChecksum,
		activeCanonicalIndices: params.activeMask.activeCanonicalIndices,
		signature: params.activeMask.signature,
		surfaceFlagsByActiveCell: params.surfaceFlagsByActiveCell
	};
}

export function buildClassifiedAnalysisActiveMask(
	params: ClassifiedActiveMaskBuildParams
): ClassifiedAnalysisActiveMask | undefined {
	const result = buildClassifiedActiveMaskSurface(params);
	if (!result) {
		return undefined;
	}
	return createClassifiedAnalysisActiveMask(result);
}
