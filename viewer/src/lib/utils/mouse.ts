/**
 * Shared Mouse Utilities
 * 
 * Generalizes mouse position normalization used across components
 */

import * as THREE from 'three';

/**
 * Get normalized mouse position from DOM event
 * @param event - Mouse event
 * @param rect - Bounding rectangle of the element
 * @returns Normalized mouse coordinates (-1 to 1)
 */
export function getNormalizedMousePosition(
	event: MouseEvent,
	rect: DOMRect
): THREE.Vector2 {
	return new THREE.Vector2(
		((event.clientX - rect.left) / rect.width) * 2 - 1,
		-((event.clientY - rect.top) / rect.height) * 2 + 1
	);
}


