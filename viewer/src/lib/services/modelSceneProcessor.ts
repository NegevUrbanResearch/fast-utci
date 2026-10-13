import { applyModelCoordinateTransform, calculateScenarioOrigin, applyModelOffset } from '$lib/utils/coordinates';
import { Group, Mesh, Vector3 } from 'three';
import type { AnalysisMetadata } from '$lib/types/analysis';
import { applyLayerMaterials } from '$lib/services/modelLoaderService';
import { initializeLayerManager, getDiscoveredLayers, resetLayerManager, applyLayerVisibilityState } from '$lib/services/layerManagerService';
import { layerStore } from '$lib/stores/layerStore';
import { get } from 'svelte/store';
import { getAnchorOffset, isNormalizationEnabled } from '$lib/config/viewerConfig';
import { cacheModel, hasModelInCache } from '$lib/services/modelCacheService';
function cloneWithMaterials(scene: Group): Group {
	const clone = scene.clone(true);
	clone.traverse((child) => {
		if (child instanceof Mesh && child.material) {
			child.material = Array.isArray(child.material)
				? child.material.map(material => material.clone()) : child.material.clone();
		}
	});
	return clone;
}
/** Process a captured load; only its current owner can publish shared state. */
export async function processModelScene(scene: Group, fromCache: boolean, snapshot: {
	modelPath: string;
	coordinateSystem: 'xy_ground' | 'xz_ground';
	metadata?: AnalysisMetadata;
}, publish: {
	layersDiscovered: (layers: string[]) => void;
	modelLoaded: (scene: Group) => void;
}, isCurrent: () => boolean, reportError: (error: unknown) => void = () => { }): Promise<Group | undefined> {
	try {
		if (!isCurrent())
			return;
		const { modelPath, coordinateSystem, metadata } = snapshot;
		const processed = fromCache ? cloneWithMaterials(scene) : scene;
		if (!fromCache) {
			await applyLayerMaterials(processed);
			if (!isCurrent())
				return;
		}
		resetLayerManager();
		initializeLayerManager(processed);
		applyLayerVisibilityState(get(layerStore));
		publish.layersDiscovered(getDiscoveredLayers());
		if (!isCurrent())
			return;
		if (!fromCache) {
			applyModelCoordinateTransform(processed, coordinateSystem);
			// Cache before normalization; cached instances share geometry but own materials.
			if (!hasModelInCache(modelPath))
				cacheModel(modelPath, cloneWithMaterials(processed));
		}
		if (isNormalizationEnabled() && metadata) {
			const origin = calculateScenarioOrigin(metadata);
			const transformedOrigin = coordinateSystem === 'xy_ground'
				? new Vector3(origin.x, origin.z, -origin.y) : origin.clone();
			const offset = getAnchorOffset().clone().sub(transformedOrigin);
			if (offset.lengthSq() > 0.001)
				applyModelOffset(processed, offset);
		}
		if (!isCurrent())
			return;
		publish.modelLoaded(processed);
		return processed;
	}
	catch (error) {
		if (isCurrent())
			reportError(error);
	}
}
