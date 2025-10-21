/**
 * Layer Manager for Dynamic Layer Discovery and UI Generation
 * 
 * Discovers which layer types exist in a loaded model and dynamically creates
 * UI controls for layer visibility toggling.
 */

import { LAYER_MATERIALS, STANDARD_LAYER_TYPES } from '../config/layer-materials.js';

export class LayerManager {
    constructor(container, utciToggleCallback = null) {
        this.container = container;
        this.layers = new Map();  // layerType -> { visible, meshes, config }
        this.utciToggleCallback = utciToggleCallback;  // Callback for UTCI toggle
    }
    
    /**
     * Discover which layer types exist in the loaded model
     * 
     * @param {THREE.Group} model - Loaded model to scan
     */
    discoverLayers(model) {
        this.layers.clear();
        
        model.traverse((child) => {
            if (child.isMesh && child.userData.layerType) {
                const layerType = child.userData.layerType;
                
                if (!this.layers.has(layerType)) {
                    // Find standard layer config
                    const config = STANDARD_LAYER_TYPES.find(t => t.id === layerType);
                    
                    this.layers.set(layerType, {
                        visible: config?.defaultVisible ?? true,
                        meshes: [],
                        config: config || { 
                            id: layerType, 
                            displayName: layerType.charAt(0).toUpperCase() + layerType.slice(1) 
                        }
                    });
                }
                
                this.layers.get(layerType).meshes.push(child);
            }
        });
        
        console.log(`[LAYERS] Discovered: ${[...this.layers.keys()].join(', ')}`);
    }
    
    /**
     * Build UI controls dynamically based on discovered layers
     * 
     * Only shows toggles for layer types that exist in the current model.
     * Also includes UTCI layer toggle.
     */
    createControls() {
        this.container.innerHTML = '<div style="font-weight: bold; margin-bottom: 10px; font-size: 13px;">Model Layers</div>';
        
        // Only show toggles for types that exist in this model
        // Use STANDARD_LAYER_TYPES order for consistent UI
        STANDARD_LAYER_TYPES.forEach(layerConfig => {
            if (this.layers.has(layerConfig.id)) {
                const layer = this.layers.get(layerConfig.id);
                const toggle = this.createToggle(layerConfig, layer.visible);
                this.container.appendChild(toggle);
            }
        });
        
        // Add UTCI layer toggle (not part of the model, but part of the visualization)
        this.addUTCIToggle();
        
        console.log(`[LAYERS] Created ${this.layers.size} model layer controls + UTCI control`);
    }
    
    /**
     * Add UTCI layer toggle control
     */
    addUTCIToggle() {
        const div = document.createElement('div');
        div.className = 'layer-item active';  // UTCI visible by default
        div.dataset.layer = 'utci';
        div.dataset.visible = 'true';
        
        // Color box with UTCI color
        const colorBox = document.createElement('div');
        colorBox.className = 'layer-color utci';
        
        // Label
        const label = document.createElement('div');
        label.className = 'layer-label';
        label.textContent = 'UTCI Data';
        
        div.appendChild(colorBox);
        div.appendChild(label);
        
        // Add click handler
        div.addEventListener('click', () => {
            const currentVisible = div.dataset.visible === 'true';
            const newVisible = !currentVisible;
            div.dataset.visible = newVisible.toString();
            
            // Toggle active class
            if (newVisible) {
                div.classList.add('active');
            } else {
                div.classList.remove('active');
            }
            
            // Call callback if provided (to toggle point cloud visibility)
            if (this.utciToggleCallback) {
                this.utciToggleCallback(newVisible);
            }
            
            console.log(`[LAYER] UTCI visibility: ${newVisible}`);
        });
        
        this.container.appendChild(div);
    }
    
    /**
     * Create a single layer toggle element
     * 
     * @param {Object} layerConfig - Layer configuration
     * @param {boolean} isVisible - Initial visibility state
     * @returns {HTMLElement} Toggle element
     */
    createToggle(layerConfig, isVisible) {
        const div = document.createElement('div');
        div.className = 'layer-item' + (isVisible ? ' active' : '');
        div.dataset.layer = layerConfig.id;
        div.dataset.visible = isVisible;
        
        // Color box
        const colorBox = document.createElement('div');
        colorBox.className = 'layer-color';
        colorBox.style.backgroundColor = LAYER_MATERIALS[layerConfig.id].color;
        
        // Label
        const label = document.createElement('div');
        label.className = 'layer-label';
        label.textContent = layerConfig.displayName;
        
        div.appendChild(colorBox);
        div.appendChild(label);
        
        // Click handler to toggle visibility
        div.addEventListener('click', () => {
            const newVisible = !this.layers.get(layerConfig.id).visible;
            this.toggleLayer(layerConfig.id, newVisible);
            div.classList.toggle('active');
            div.dataset.visible = newVisible;
        });
        
        return div;
    }
    
    /**
     * Toggle visibility of a layer type
     * 
     * @param {string} layerType - Layer type to toggle
     * @param {boolean} visible - New visibility state
     */
    toggleLayer(layerType, visible) {
        const layer = this.layers.get(layerType);
        if (layer) {
            layer.visible = visible;
            layer.meshes.forEach(mesh => {
                mesh.visible = visible;
            });
            console.log(`[LAYER] ${visible ? 'Show' : 'Hide'} ${layerType}: ${layer.meshes.length} meshes`);
        }
    }
    
    /**
     * Initialize visibility based on discovered layers and default settings
     */
    initializeVisibility() {
        this.layers.forEach((layer, layerType) => {
            layer.meshes.forEach(mesh => {
                mesh.visible = layer.visible;
            });
        });
        
        console.log('[LAYERS] Initialized visibility for all layers');
    }
    
    /**
     * Get all layer types present in the model
     * 
     * @returns {Array<string>} Array of layer type IDs
     */
    getLayerTypes() {
        return [...this.layers.keys()];
    }
    
    /**
     * Get mesh count for a layer type
     * 
     * @param {string} layerType - Layer type
     * @returns {number} Number of meshes in this layer
     */
    getMeshCount(layerType) {
        const layer = this.layers.get(layerType);
        return layer ? layer.meshes.length : 0;
    }
    
    /**
     * Check if a layer type is currently visible
     * 
     * @param {string} layerType - Layer type
     * @returns {boolean} True if layer is visible
     */
    isLayerVisible(layerType) {
        const layer = this.layers.get(layerType);
        return layer ? layer.visible : false;
    }
}

