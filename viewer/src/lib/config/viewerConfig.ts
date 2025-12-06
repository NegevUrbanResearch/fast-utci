/**
 * Viewer Configuration
 *
 * Runtime settings for model alignment and normalization.
 */
import * as THREE from 'three';

export interface ViewerConfig {
	/** Anchor point for model alignment */
	anchorOffset: THREE.Vector3;
	/** Enable runtime normalization (translate models/analysis to anchor) */
	enableNormalization: boolean;
}

const defaultConfig: ViewerConfig = {
	anchorOffset: new THREE.Vector3(0, 0, 0),
	enableNormalization: true
};

let currentConfig: ViewerConfig = { ...defaultConfig };

export function getAnchorOffset(): THREE.Vector3 {
	return currentConfig.anchorOffset.clone();
}

export function isNormalizationEnabled(): boolean {
	return currentConfig.enableNormalization;
}

export function updateViewerConfig(config: Partial<ViewerConfig>): void {
	if (config.anchorOffset) {
		currentConfig.anchorOffset = config.anchorOffset.clone();
	}
	if (config.enableNormalization !== undefined) {
		currentConfig.enableNormalization = config.enableNormalization;
	}
}

