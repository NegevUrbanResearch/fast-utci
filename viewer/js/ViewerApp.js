/**
 * Main Viewer Application
 * 
 * Orchestrates the 3D UTCI viewer with model loading, data visualization,
 * time controls, and analytics.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadAnalysis } from './UTCIDataLoader.js';
import { loadModelWithLayers, toggleLayerVisibility } from './ModelLoader.js';
import { createUTCIPointCloud, updatePointCloudColors, createRaycaster, findNearestPoint, createHighlightSphere, createColorLegend } from './UTCIRenderer.js';
import { createTimeControls } from './TimeController.js';
import { loadValidationData, compareWithValidation, createAnalyticsPanel, updateAnalyticsPanel } from './Analytics.js';
import { calculateStatistics } from './UTCIDataLoader.js';
import { getUTCILabel } from './ColorScale.js';
import { createSunPath, updateCurrentSunIndicator, createSunMarkerRaycaster, findClickedSunMarker } from './SunPathRenderer.js';

/**
 * Main Viewer App Class
 */
export class ViewerApp {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointCloud = null;
        this.analysis = null;
        this.validation = null;
        this.currentHour = 0;
        this.highlightSphere = null;
        this.colorMode = 'normalized'; // 'normalized' or 'discrete'
        this.sunPathGroup = null;
        this.sunPathVisible = false;
        this.sunTooltip = null;
        
        this.init();
    }
    
    /**
     * Initialize Three.js scene
     */
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xacd8eb); // Dark gray
        
        // Create camera with reasonable near/far planes
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 5000);
        this.camera.position.set(-2000, 300, -400);
        this.camera.lookAt(new THREE.Vector3(-2000, 0, -400));
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            powerPreference: "high-performance", // Prefer performance
            logarithmicDepthBuffer: true // Improve depth precision for large scenes
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // Disable tone mapping to show true colors (fixes grey buildings)
        this.renderer.toneMapping = THREE.NoToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        // Enable shadows with performance settings
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.BasicShadowMap; // Faster than PCF
        
        this.container.appendChild(this.renderer.domElement);
        
        // Create orbit controls with standard Three.js configuration
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        
        // Standard damping for smooth camera movement
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Zoom limits - moderate range (10:1 ratio for smoother feel)
        this.controls.minDistance = 200;
        this.controls.maxDistance = 2000;
        
        // CRITICAL: Reduce zoom sensitivity to prevent jumping to min/max
        // Mouse wheels report different deltaY values - this normalizes the behavior
        this.controls.zoomSpeed = 0.1;
        
        // Pan and rotate settings
        this.controls.screenSpacePanning = false;
        this.controls.maxPolarAngle = Math.PI / 2;
        
        // Add lights
        this.addLights();
        
        // Grid helper will be added dynamically after model is loaded
        this.gridHelper = null;
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Handle mouse events
        this.container.addEventListener('click', (e) => this.onMouseClick(e));
        this.container.addEventListener('mousemove', (e) => this.onMouseMove(e));
        
        // Custom wheel handler to clamp extreme deltaY values
        this.setupNormalizedZoom();
        
        // Setup layer control event handlers
        this.setupLayerControls();
        
        // Start animation loop
        this.animate();
        
        console.log('[OK] Viewer initialized');
    }
    
    /**
     * Setup normalized zoom to handle extreme mouse wheel deltaY values
     */
    setupNormalizedZoom() {
        // Disable default OrbitControls zoom
        this.controls.enableZoom = false;
        
        // Add custom wheel handler with normalized deltaY
        this.renderer.domElement.addEventListener('wheel', (event) => {
            event.preventDefault();
            
            // Clamp deltaY to a maximum of ±100 to normalize across different mice
            const clampedDelta = Math.max(-100, Math.min(100, event.deltaY));
            
            // Calculate zoom scale (small increments for smooth zooming)
            const scale = 1 + (clampedDelta / 1000); // 10% max per tick
            
            // Get current distance from camera to target
            const distance = this.camera.position.distanceTo(this.controls.target);
            
            // Calculate new distance
            let newDistance = distance * scale;
            
            // Clamp to min/max distance
            newDistance = Math.max(
                this.controls.minDistance,
                Math.min(this.controls.maxDistance, newDistance)
            );
            
            // Move camera along the view direction
            const direction = new THREE.Vector3()
                .subVectors(this.camera.position, this.controls.target)
                .normalize();
            
            this.camera.position.copy(this.controls.target)
                .add(direction.multiplyScalar(newDistance));
            
            this.controls.update();
            
        }, { passive: false });
        
        console.log('[OK] Normalized zoom enabled');
    }
    
    /**
     * Add dynamic grid helper based on model and UTCI data bounds
     */
    addDynamicGrid(model, analysis) {
        // Remove existing grid if present
        if (this.gridHelper) {
            this.scene.remove(this.gridHelper);
        }
        
        // Calculate bounds from model
        const modelBox = new THREE.Box3().setFromObject(model);
        const groundLevel = modelBox.min.y;
        
        // Calculate size - use the larger dimension of model + minimal padding
        const modelSize = modelBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(modelSize.x, modelSize.z);
        const gridSize = Math.ceil(maxDim * 1.2 / 100) * 100; // Round up to nearest 100, add 20% padding
        
        // Calculate number of divisions based on size
        const divisions = Math.min(50, Math.max(20, Math.floor(gridSize / 50)));
        
        // Create grid at ground level (hidden by default for cleaner look)
        this.gridHelper = new THREE.GridHelper(gridSize, divisions, 0x444444, 0x888888);
        this.gridHelper.position.y = groundLevel;
        this.gridHelper.visible = false; // Hidden by default
        
        // Center grid on model center
        const modelCenter = modelBox.getCenter(new THREE.Vector3());
        this.gridHelper.position.x = modelCenter.x;
        this.gridHelper.position.z = modelCenter.z;
        
        this.scene.add(this.gridHelper);
        
        console.log(`[GRID] Created ${gridSize}x${gridSize} grid at Y=${groundLevel.toFixed(2)}, centered at (${modelCenter.x.toFixed(2)}, ${modelCenter.z.toFixed(2)})`);
    }
    
    /**
     * Add scene lights
     */
    addLights() {
        // Very bright ambient light to ensure white appears as white
        const ambient = new THREE.AmbientLight(0xffffff, 1.2);
        this.scene.add(ambient);
        
        // Directional light (sun) - reduced since ambient is bright
        const sun = new THREE.DirectionalLight(0xffffff, 0.6);
        sun.position.set(100, 200, 100);
        sun.castShadow = true;
        sun.shadow.camera.left = -500;
        sun.shadow.camera.right = 500;
        sun.shadow.camera.top = 500;
        sun.shadow.camera.bottom = -500;
        // Reduced shadow map size for better performance
        sun.shadow.mapSize.width = 512;
        sun.shadow.mapSize.height = 512;
        this.scene.add(sun);
    }
    
    /**
     * Load and display analysis
     */
    async loadAnalysisData(analysisId) {
        try {
            // Show loading indicator
            this.showLoading('Loading analysis data...');
            
            // Load analysis data
            this.analysis = await loadAnalysis(analysisId);
            
            // Update loading message
            this.showLoading('Loading 3D model...');
            
            // Load 3D model
            const modelPath = this.analysis.metadata.model_file.replace('data/', '../data/');
            const model = await loadModelWithLayers(modelPath);
            this.model = model; // Store reference for layer controls
            this.scene.add(model);
            
            // Initialize layer visibility based on HTML defaults (roads and base hidden)
            this.initializeLayerVisibility();
            
            // Create UTCI point cloud (pass model for coordinate alignment)
            this.showLoading('Creating UTCI visualization...');
            this.pointCloud = createUTCIPointCloud(this.analysis, model, this.currentHour);
            this.scene.add(this.pointCloud);
            
            // Add dynamic grid based on model bounds
            this.addDynamicGrid(model, this.analysis);
            
            // Create sun path if full day analysis and sun data available
            if (this.analysis.metadata.analysis_type === 'full_day' && 
                this.analysis.metadata.sun_positions) {
                const modelBox = new THREE.Box3().setFromObject(model);
                const modelCenter = modelBox.getCenter(new THREE.Vector3());
                const modelSize = modelBox.getSize(new THREE.Vector3()).length();
                
                this.sunPathGroup = createSunPath(
                    this.analysis.metadata, 
                    modelCenter, 
                    modelSize
                );
                this.scene.add(this.sunPathGroup);
                
                // Initialize with current hour highlighted
                updateCurrentSunIndicator(this.sunPathGroup, this.currentHour);
            }
            
            // Add color legend
            const legend = createColorLegend(
                this.analysis.metadata.utci_range.min,
                this.analysis.metadata.utci_range.max,
                this.analysis.metadata.analysis_type
            );
            document.body.appendChild(legend);
            
            // Setup color mode switch if full day
            if (this.analysis.metadata.analysis_type === 'full_day') {
                const switchEl = document.getElementById('color-mode-switch');
                if (switchEl && typeof switchEl.addEventListener === 'function') {
                    // Initialize: checked => Per Hour (discrete), unchecked => Full Day (normalized)
                    switchEl.checked = this.colorMode === 'discrete';
                    switchEl.addEventListener('sl-change', () => {
                        this.colorMode = switchEl.checked ? 'discrete' : 'normalized';
                        updatePointCloudColors(this.pointCloud, this.analysis, this.currentHour, this.colorMode);
                        this.updateLegendRange();
                    });
                } else {
                    // Fallback: if sl-switch not available yet, listen for native change
                    const fallback = document.getElementById('color-mode-switch');
                    if (fallback) {
                        fallback.addEventListener('change', () => {
                            this.colorMode = fallback.checked ? 'discrete' : 'normalized';
                            updatePointCloudColors(this.pointCloud, this.analysis, this.currentHour, this.colorMode);
                            this.updateLegendRange();
                        });
                    }
                }
            }
            
            // Add time controls if full day analysis
            if (this.analysis.metadata.analysis_type === 'full_day') {
                const timeControls = createTimeControls(
                    this.analysis.metadata.hours.length,
                    (hourIndex) => this.onHourChange(hourIndex)
                );
                document.body.appendChild(timeControls);
            }
            
            // Try to load validation data and create analytics panel
            try {
                this.validation = await loadValidationData();
                const comparisonStats = compareWithValidation(this.analysis, this.validation, this.currentHour);
                const analyticsPanel = createAnalyticsPanel(this.analysis.metadata, comparisonStats);
                document.body.appendChild(analyticsPanel);
            } catch (error) {
                console.warn('[WARN] Could not load validation data:', error);
                // Create panel without comparison
                const analyticsPanel = createAnalyticsPanel(this.analysis.metadata);
                document.body.appendChild(analyticsPanel);
            }
            
            // Setup sun path toggle
            const sunPathToggle = document.getElementById('sun-path-toggle');
            if (sunPathToggle && this.sunPathGroup) {
                sunPathToggle.addEventListener('change', (e) => {
                    this.sunPathGroup.visible = e.target.checked;
                    this.sunPathVisible = e.target.checked;
                });
            }
            
            // Hide loading indicator
            this.hideLoading();
            
            // Focus camera on model
            this.focusCameraOnModel(model);
            
            console.log('[OK] Analysis loaded successfully');
            
        } catch (error) {
            console.error('[ERROR] Failed to load analysis:', error);
            this.showError(`Failed to load analysis: ${error.message}`);
        }
    }
    
    /**
     * Handle hour change (full day analysis)
     */
    onHourChange(hourIndex) {
        if (!this.pointCloud || !this.analysis) return;
        
        this.currentHour = hourIndex;
        updatePointCloudColors(this.pointCloud, this.analysis, hourIndex, this.colorMode);
        
        // Update legend if in discrete mode (both text and gradient labels)
        if (this.colorMode === 'discrete') {
            this.updateLegendRange();
        }
        
        // Update sun path indicator
        if (this.sunPathGroup) {
            updateCurrentSunIndicator(this.sunPathGroup, hourIndex);
        }
        
        // Update analytics if validation data is available
        if (this.validation) {
            const comparisonStats = compareWithValidation(this.analysis, this.validation, hourIndex);
            // For full day analysis, also pass current analysis stats for dynamic UTCI range
            const analysisStats = this.analysis.data.numHours > 1 ? 
                calculateStatistics(this.analysis.data.utciByHour[hourIndex]) : null;
            updateAnalyticsPanel(comparisonStats, analysisStats);
        }
    }
    
    /**
     * Toggle color mode between normalized and discrete
     */
    toggleColorMode() {
        this.colorMode = this.colorMode === 'normalized' ? 'discrete' : 'normalized';
        
        // Update button label to show CURRENT state (not next state)
        const label = document.getElementById('color-mode-label');
        if (label) {
            // Show what mode is NOW active
            label.textContent = this.colorMode === 'normalized' ? 'Full Day' : 'Per Hour';
        }
        
        // Update colors
        updatePointCloudColors(this.pointCloud, this.analysis, this.currentHour, this.colorMode);
        
        // Update legend range display
        this.updateLegendRange();
    }
    
    /**
     * Update legend range display based on current color mode
     */
    updateLegendRange() {
        const rangeText = document.getElementById('legend-range-text');
        if (!rangeText || !this.analysis) return;
        
        let min, max, modeLabel;
        if (this.colorMode === 'normalized') {
            min = this.analysis.metadata.utci_range.min;
            max = this.analysis.metadata.utci_range.max;
            modeLabel = '';
        } else {
            const hourStat = this.analysis.metadata.hour_statistics[this.currentHour];
            min = hourStat.min;
            max = hourStat.max;
            const hour = this.analysis.metadata.hours[this.currentHour];
            modeLabel = ` (Hour ${hour}:00)`;
        }
        
        // Update range text
        rangeText.innerHTML = `Range: ${min.toFixed(1)} - ${max.toFixed(1)}°C${modeLabel}`;
        
        // Update gradient scale labels
        this.updateLegendLabels(min, max);
    }
    
    /**
     * Update legend gradient scale labels
     */
    updateLegendLabels(utciMin, utciMax) {
        const legend = document.getElementById('utci-legend');
        if (!legend) return;
        
        // Find all temperature label divs in the legend
        const labelContainer = legend.querySelector('[style*="position: relative; flex: 1"]');
        if (!labelContainer) return;
        
        // Clear existing labels
        labelContainer.innerHTML = '';
        
        // Recreate labels with new range
        const tempRange = utciMax - utciMin;
        const numLabels = 6;
        const tempStep = tempRange / (numLabels - 1);
        
        for (let i = 0; i < numLabels; i++) {
            const temp = utciMax - (i * tempStep);  // Start from max (top) to min (bottom)
            const position = (i / (numLabels - 1)) * 100;
            
            const labelDiv = document.createElement('div');
            labelDiv.style.cssText = `
                position: absolute;
                left: 8px;
                top: ${position}%;
                transform: translateY(-50%);
                font-size: 12px;
                font-weight: 500;
                white-space: nowrap;
            `;
            labelDiv.textContent = `${temp.toFixed(1)}°C`;
            labelContainer.appendChild(labelDiv);
        }
    }
    
    /**
     * Handle mouse click for point selection
     */
    onMouseClick(event) {
        // Check if sun path is visible and user clicked a marker
        if (this.sunPathVisible && this.sunPathGroup) {
            const mouse = this.getMousePosition(event);
            const raycaster = createSunMarkerRaycaster(this.camera, mouse);
            const clickedMarker = findClickedSunMarker(this.sunPathGroup, raycaster);
            
            if (clickedMarker) {
                // Jump to clicked hour
                const slider = document.getElementById('hour-slider');
                if (slider) {
                    slider.value = clickedMarker.hour;
                    slider.dispatchEvent(new Event('input'));
                }
                return; // Consumed click
            }
        }
        
        // Click functionality disabled for UTCI points - using hover tooltip instead
        // Optional: could add different click behavior here if needed
    }
    
    /**
     * Handle mouse move for hover effects
     */
    onMouseMove(event) {
        // Check sun path markers first if visible
        if (this.sunPathVisible && this.sunPathGroup) {
            const mouse = this.getMousePosition(event);
            const raycaster = createSunMarkerRaycaster(this.camera, mouse);
            const hoveredMarker = findClickedSunMarker(this.sunPathGroup, raycaster);
            
            if (hoveredMarker) {
                this.showSunTooltip(hoveredMarker, event.clientX, event.clientY);
                this.hideTooltip(); // Hide UTCI tooltip
                return; // Don't check UTCI points
            } else {
                this.hideSunTooltip();
            }
        }
        
        // Check UTCI points
        if (!this.pointCloud || !this.pointCloud.visible) {
            this.hideTooltip();
            this.hideSunTooltip();
            return;
        }
        
        const mouse = this.getMousePosition(event);
        // Pass grid size from analysis metadata for accurate point picking
        const gridSize = this.analysis?.metadata?.grid_size || 10.0;
        const raycaster = createRaycaster(this.camera, mouse, gridSize);
        const intersection = findNearestPoint(this.pointCloud, raycaster);
        
        // Show tooltip if a point is found
        // The raycaster threshold is dynamically set based on grid size
        if (intersection) {
            this.showTooltip(intersection.index, event.clientX, event.clientY);
        } else {
            this.hideTooltip();
        }
    }
    
    /**
     * Show information about selected point
     */
    showPointInfo(pointIndex) {
        if (!this.analysis) return;
        
        const position = {
            x: this.analysis.data.positions[pointIndex * 3],
            y: this.analysis.data.positions[pointIndex * 3 + 1],
            z: this.analysis.data.positions[pointIndex * 3 + 2]
        };
        
        let utciValue;
        if (this.analysis.data.numHours === 1) {
            utciValue = this.analysis.data.utciValues[pointIndex];
        } else {
            utciValue = this.analysis.data.utciByHour[this.currentHour][pointIndex];
        }
        
        const label = getUTCILabel(utciValue, false);
        
        // Create or update info panel
        let infoPanel = document.getElementById('point-info');
        if (!infoPanel) {
            infoPanel = document.createElement('div');
            infoPanel.id = 'point-info';
            infoPanel.style.cssText = `
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif;
                font-size: 13px;
                z-index: 100;
            `;
            document.body.appendChild(infoPanel);
        }
        
        infoPanel.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 8px;">Selected Point</div>
            <div><strong>Position:</strong><br>
                X: ${position.x.toFixed(1)}m<br>
                Y: ${position.y.toFixed(1)}m<br>
                Z: ${position.z.toFixed(1)}m
            </div>
            <div style="margin-top: 8px;">
                <strong>UTCI:</strong> ${utciValue.toFixed(1)}°C<br>
                <strong>Category:</strong> ${label}
            </div>
        `;
    }
    
    /**
     * Highlight selected point
     */
    highlightPoint(position) {
        // Remove previous highlight
        if (this.highlightSphere) {
            this.scene.remove(this.highlightSphere);
        }
        
        // Create new highlight
        this.highlightSphere = createHighlightSphere(position, 2.0);
        this.scene.add(this.highlightSphere);
    }
    
    /**
     * Show hover tooltip with UTCI data
     */
    showTooltip(pointIndex, x, y) {
        if (!this.analysis || !this.pointCloud) return;
        
        // Create tooltip if it doesn't exist
        if (!this.tooltip) {
            this.tooltip = document.createElement('div');
            this.tooltip.id = 'utci-tooltip';
            this.tooltip.style.cssText = `
                position: fixed;
                background: rgba(0, 0, 0, 0.85);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                line-height: 1.5;
                display: none;
            `;
            document.body.appendChild(this.tooltip);
        }
        
        // Get position from the actual point cloud geometry (transformed positions)
        const position = new THREE.Vector3();
        position.fromBufferAttribute(
            this.pointCloud.geometry.attributes.position,
            pointIndex
        );
        
        // Get UTCI value
        let utciValue;
        if (this.analysis.data.numHours === 1) {
            utciValue = this.analysis.data.utciValues[pointIndex];
        } else {
            utciValue = this.analysis.data.utciByHour[this.currentHour][pointIndex];
        }
        
        // Update tooltip content
        this.tooltip.innerHTML = `
            <strong>UTCI:</strong> ${utciValue.toFixed(1)}°C<br>
            <strong>Position:</strong><br>
            X: ${position.x.toFixed(1)}m, Y: ${position.y.toFixed(1)}m, Z: ${position.z.toFixed(1)}m
        `;
        
        // Position tooltip near mouse (offset to avoid cursor overlap)
        this.tooltip.style.left = (x + 15) + 'px';
        this.tooltip.style.top = (y + 15) + 'px';
        this.tooltip.style.display = 'block';
    }
    
    /**
     * Hide hover tooltip
     */
    hideTooltip() {
        if (this.tooltip) {
            this.tooltip.style.display = 'none';
        }
    }
    
    /**
     * Show sun marker tooltip with sun position data
     */
    showSunTooltip(markerData, x, y) {
        if (!this.sunTooltip) {
            this.sunTooltip = document.createElement('div');
            this.sunTooltip.id = 'sun-tooltip';
            this.sunTooltip.style.cssText = `
                position: fixed;
                background: rgba(255, 150, 0, 0.95);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                pointer-events: none;
                z-index: 1000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                line-height: 1.5;
                display: none;
            `;
            document.body.appendChild(this.sunTooltip);
        }
        
        const isDay = markerData.sunData.is_up;
        this.sunTooltip.innerHTML = `
            <strong>Hour:</strong> ${markerData.hour}:00<br>
            <strong>Altitude:</strong> ${markerData.sunData.altitude.toFixed(1)}°<br>
            <strong>Azimuth:</strong> ${markerData.sunData.azimuth.toFixed(1)}°<br>
            <strong>Status:</strong> ${isDay ? '☀️ Day' : '🌙 Night'}
        `;
        
        this.sunTooltip.style.left = (x + 15) + 'px';
        this.sunTooltip.style.top = (y + 15) + 'px';
        this.sunTooltip.style.display = 'block';
    }
    
    /**
     * Hide sun marker tooltip
     */
    hideSunTooltip() {
        if (this.sunTooltip) {
            this.sunTooltip.style.display = 'none';
        }
    }
    
    /**
     * Get normalized mouse position
     */
    getMousePosition(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        return new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        );
    }
    
    /**
     * Focus camera on model
     */
    focusCameraOnModel(model) {
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        // Calculate ideal camera distance to fit the model
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraDistance = Math.abs(maxDim / 2 / Math.tan(fov / 2));
        cameraDistance *= 1.5; // Add some margin
        
        console.log(`[CAMERA] Model size: ${maxDim.toFixed(1)}m, Camera distance: ${cameraDistance.toFixed(1)}m`);
        
        // Improve depth precision: tighten near/far planes to scene scale
        const near = Math.max(0.5, maxDim / 1000);
        const far = Math.max(maxDim * 4, near + 10);
        this.camera.near = near;
        this.camera.far = far;
        this.camera.updateProjectionMatrix();
        console.log(`[CAMERA] Near/Far set to ${near.toFixed(3)} / ${far.toFixed(1)}`);

        // Position camera at an angle for better perspective
        this.camera.position.set(
            center.x + cameraDistance * 0.7, 
            center.y + cameraDistance * 0.5, 
            center.z + cameraDistance * 0.7
        );
        this.camera.lookAt(center);
        this.controls.target.copy(center);
        this.controls.update();
    }
    
    /**
     * Show loading indicator
     */
    showLoading(message) {
        let loader = document.getElementById('loading-indicator');
        if (!loader) {
            loader = document.createElement('div');
            loader.id = 'loading-indicator';
            loader.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif;
                font-size: 16px;
                z-index: 1000;
            `;
            document.body.appendChild(loader);
        }
        loader.textContent = message;
        loader.style.display = 'block';
    }
    
    /**
     * Hide loading indicator
     */
    hideLoading() {
        const loader = document.getElementById('loading-indicator');
        if (loader) {
            loader.style.display = 'none';
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        this.hideLoading();
        alert(message);
    }
    
    /**
     * Initialize layer visibility based on HTML defaults
     */
    initializeLayerVisibility() {
        const layerItems = document.querySelectorAll('.layer-item');
        
        layerItems.forEach(item => {
            const layerType = item.getAttribute('data-layer');
            const isVisible = item.getAttribute('data-visible') === 'true';
            
            // Skip UTCI layer as it's handled separately
            if (layerType === 'utci') return;
            
            // Apply visibility to 3D objects
            if (this.model) {
                toggleLayerVisibility(this.model, layerType, isVisible);
                console.log(`[INIT] ${layerType} layer visibility: ${isVisible}`);
            }
            
            // Update visual state of layer control
            if (isVisible) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }
    
    /**
     * Setup layer control event handlers
     */
    setupLayerControls() {
        const layerItems = document.querySelectorAll('.layer-item');
        
        layerItems.forEach(item => {
            item.addEventListener('click', () => {
                const layerType = item.getAttribute('data-layer');
                
                // Toggle visibility state
                const currentVisible = item.getAttribute('data-visible') === 'true';
                const newVisible = !currentVisible;
                item.setAttribute('data-visible', newVisible.toString());
                
                // Toggle 'active' class for visual feedback
                if (newVisible) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
                
                // Toggle layer in scene
                if (layerType === 'utci') {
                    // Toggle UTCI point cloud visibility
                    if (this.pointCloud) {
                        this.pointCloud.visible = newVisible;
                        console.log(`[LAYER] UTCI visibility: ${newVisible}`);
                    }
                } else {
                    // Toggle model layers
                    if (this.model) {
                        toggleLayerVisibility(this.model, layerType, newVisible);
                    }
                }
            });
        });
    }
    
    /**
     * Handle window resize
     */
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

/**
 * Initialize viewer from URL parameters
 */
export function initFromURL() {
    const params = new URLSearchParams(window.location.search);
    const analysisId = params.get('analysis');
    
    if (!analysisId) {
        alert('No analysis specified. Please use: viewer.html?analysis=<analysis_id>');
        return;
    }
    
    const container = document.getElementById('viewer-container');
    if (!container) {
        console.error('[ERROR] Container element not found');
        return;
    }
    
    const app = new ViewerApp(container);
    app.loadAnalysisData(analysisId);
}
