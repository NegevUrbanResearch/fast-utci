import * as THREE from 'three';

export type MainRouteModelLoadedEffects = {
	sceneBounds: THREE.Box3;
	cameraFit?: {
		position: THREE.Vector3;
		target: THREE.Vector3;
	};
	nextHasFitOnce: boolean;
};

// Task 3 wires these values directly from route-time model calculations, so keep
// the helper pure and explicit rather than recomputing from one source.
export function getMainRouteModelLoadedEffects(params: {
	bounds: THREE.Box3;
	center: THREE.Vector3;
	size: THREE.Vector3;
	hasFitOnce: boolean;
}): MainRouteModelLoadedEffects {
	if (params.hasFitOnce) {
		return {
			sceneBounds: params.bounds.clone(),
			nextHasFitOnce: true
		};
	}

	const maxDim = Math.max(params.size.x, params.size.y, params.size.z);
	const distance = maxDim * 1.05;
	const position = params.center
		.clone()
		.add(new THREE.Vector3(0, distance, 0.01));

	return {
		sceneBounds: params.bounds.clone(),
		cameraFit: {
			position,
			target: params.center.clone()
		},
		nextHasFitOnce: true
	};
}
