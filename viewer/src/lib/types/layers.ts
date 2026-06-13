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
	renderOrder?: number;
	outlineOnly?: boolean;
	outlineColor?: string;
	outlineOpacity?: number;
	outlineThresholdAngle?: number;
	outlineDepthTest?: boolean;
	outlineDepthWrite?: boolean;
	outlineToneMapped?: boolean;
}

/**
 * Standard layer type definition
 */
export interface StandardLayerType {
	id: string;
	displayName: string;
	defaultVisible: boolean;
}


