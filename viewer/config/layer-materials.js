/**
 * Layer Material Definitions and Name Mappings
 * 
 * This configuration file defines:
 * 1. Standard layer types with visual properties (colors, opacity, etc.)
 * 2. Mapping from actual GLB layer names to standard types
 * 3. UI display order and default visibility settings
 */

// Standard layer types with visual properties
export const LAYER_MATERIALS = {
    building: {
        color: '#ffffff',
        opacity: 1.0,
        displayName: 'Buildings',
        emissive: '#ffffff',
        emissiveIntensity: 0.3,
        materialType: 'lambert'  // MeshLambertMaterial for better performance
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
        displayName: 'Vegetation',
        materialType: 'standard'
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
export const LAYER_NAME_MAPPING = {
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
    
    // Vegetation variants
    'treesurface': 'vegetation',
    'trees': 'vegetation',
    'vegetation': 'vegetation',
    'plants': 'vegetation',
    'greenery': 'vegetation',
    
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
export const STANDARD_LAYER_TYPES = [
    { id: 'building', displayName: 'Buildings', defaultVisible: true },
    { id: 'vegetation', displayName: 'Vegetation', defaultVisible: true },
    { id: 'road', displayName: 'Roads', defaultVisible: false },
    { id: 'sidewalk', displayName: 'Sidewalks', defaultVisible: false },
    { id: 'base', displayName: 'Ground', defaultVisible: false },
    { id: 'water', displayName: 'Water', defaultVisible: true }
];

