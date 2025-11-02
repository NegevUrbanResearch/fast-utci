/**
 * Type definitions for 3D model layer configurations
 */

/**
 * Material type for Three.js
 */
export type MaterialType = 'lambert' | 'standard';

/**
 * Layer material configuration
 */
export interface LayerMaterialConfig {
	color: string;
	opacity: number;
	displayName: string;
	materialType: MaterialType;
	emissive?: string;
	emissiveIntensity?: number;
	polygonOffset?: boolean;
}

/**
 * Standard layer type definition
 */
export interface StandardLayerType {
	id: string;
	displayName: string;
	defaultVisible: boolean;
}


