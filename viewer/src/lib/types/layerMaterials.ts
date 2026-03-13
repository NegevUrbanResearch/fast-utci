/**
 * Layer Material Definitions and Name Mappings
 * 
 * This configuration file defines:
 * 1. Standard layer types with visual properties (colors, opacity, etc.)
 * 2. Mapping from actual GLB layer names to standard types
 * 3. UI display order and default visibility settings
 */

import type { LayerMaterialConfig, StandardLayerType } from './layers';

// Standard layer types with visual properties
export const LAYER_MATERIALS: Record<string, LayerMaterialConfig> = {
	building: {
		color: '#ffffff',
		opacity: 1.0,
		displayName: 'Buildings',
		emissive: '#ffffff',
		emissiveIntensity: 0.3,
		materialType: 'lambert'  // MeshLambertMaterial for better performance
	},
	new_building: {
		color: '#93a6a6',  // Slightly blue-tinted white
		opacity: 1.0,
		displayName: 'New Buildings',
		emissive: '#e8f4f8',
		emissiveIntensity: 0.35,
		materialType: 'lambert'
	},
	new_vegetation: {
		color: '#38ff8c',  // Brighter green than existing vegetation
		opacity: 0.9,
		displayName: 'New Trees',
		materialType: 'standard',
		polygonOffset: true
	},
	base: {
		color: '#BDC3C7',
		opacity: 0.2,
		displayName: 'Ground',
		polygonOffset: true,  // Prevent z-fighting with UTCI overlay
		materialType: 'standard'
	},
	road: {
		color: '#34495E',
		opacity: 0.7,
		displayName: 'Roads',
		materialType: 'standard'
	},
	sidewalk: {
		color: '#95a5a6',
		opacity: 0.8,
		displayName: 'Sidewalks',
		materialType: 'standard'
	},
	vegetation: {
		color: '#27AE60',
		opacity: 0.9,
		displayName: 'Trees',
		materialType: 'standard',
		polygonOffset: true
	},
	water: {
		color: '#3498DB',
		opacity: 0.6,
		displayName: 'Water',
		materialType: 'standard'
	},
	default: {
		color: '#95a5a6',
		opacity: 0.8,
		displayName: 'Other',
		materialType: 'standard'
	}
};

// Map actual GLB layer names (from scene graph) to standard types
// This handles variations in naming conventions from different modeling software
export const LAYER_NAME_MAPPING: Record<string, string> = {
	// Ground/Base variants
	'ground': 'base',
	'terrain': 'base',
	'base': 'base',
	'floor': 'base',
	
	// Road variants
	'roads': 'road',
	'road': 'road',
	'street': 'road',
	'streets': 'road',
	'highway': 'road',
	
	// Building variants
	'building': 'building',
	'buildings': 'building',
	'existing_building - copy': 'building',
	'structure': 'building',
	'facade': 'building',
	
	// New building variants
	'new building': 'new_building',
	'new buildings': 'new_building',
	'new_building': 'new_building',
	'new_buildings': 'new_building',
	'proposed building': 'new_building',
	'proposed buildings': 'new_building',
	
	// Vegetation variants
	'treesurface': 'vegetation',
	'trees': 'vegetation',
	'vegetation': 'vegetation',
	'plants': 'vegetation',
	'greenery': 'vegetation',
	
	// New vegetation variants
	'new trees': 'new_vegetation',
	'new tree': 'new_vegetation',
	'new_trees': 'new_vegetation',
	'new_tree': 'new_vegetation',
	'proposed trees': 'new_vegetation',
	'proposed vegetation': 'new_vegetation',
	
	// Sidewalk/Parking variants
	'sidewalk': 'sidewalk',
	'sidewalks': 'sidewalk',
	'added sidewalk and parking': 'sidewalk',
	'parking': 'sidewalk',
	'footpath': 'sidewalk',
	'walkway': 'sidewalk',
	
	// Water variants
	'water': 'water',
	'pond': 'water',
	'lake': 'water',
	'river': 'water'
};

// Standard layer types for UI controls
// Defines display order and default visibility
export const STANDARD_LAYER_TYPES: StandardLayerType[] = [
	{ id: 'building', displayName: 'Buildings', defaultVisible: true },
	{ id: 'new_building', displayName: 'New Buildings', defaultVisible: true },
	{ id: 'vegetation', displayName: 'Trees', defaultVisible: true },
	{ id: 'new_vegetation', displayName: 'New Trees', defaultVisible: true },
	{ id: 'road', displayName: 'Roads', defaultVisible: false },
	{ id: 'sidewalk', displayName: 'Sidewalks', defaultVisible: false },
	{ id: 'base', displayName: 'Ground', defaultVisible: false },
	{ id: 'water', displayName: 'Water', defaultVisible: true },
	{ id: 'unknown', displayName: 'Unknown', defaultVisible: false }
];


