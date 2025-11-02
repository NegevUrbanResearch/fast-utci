import { describe, it, expect, beforeEach, vi } from 'vitest';
import { get } from 'svelte/store';
import { layerStore, toggleLayer, setLayerVisible, discoveredLayersStore, setDiscoveredLayers } from '$lib/stores/layerStore';
import * as layerManagerService from '$lib/services/layerManagerService';

// Mock the layerManagerService
vi.mock('$lib/services/layerManagerService', () => ({
	toggleLayerVisibility: vi.fn(),
	initializeLayerManager: vi.fn(),
	resetLayerManager: vi.fn(),
	getDiscoveredLayers: vi.fn(() => []),
	applyLayerVisibilityState: vi.fn()
}));

describe('layerStore', () => {
	beforeEach(() => {
		// Reset store before each test
		layerStore.set({
			building: true,
			new_building: true,
			vegetation: true,
			new_vegetation: true,
			road: false,
			sidewalk: false,
			base: false,
			water: true
		});
		
		discoveredLayersStore.set([]);
		vi.clearAllMocks();
	});

	it('should initialize with default visibility values', () => {
		const store = get(layerStore);
		expect(store.building).toBe(true);
		expect(store.vegetation).toBe(true);
		expect(store.road).toBe(false);
	});

	it('should toggle layer visibility in store', () => {
		toggleLayer('building');
		const store = get(layerStore);
		expect(store.building).toBe(false);

		toggleLayer('building');
		expect(get(layerStore).building).toBe(true);
	});

	it('should call layerManagerService.toggleLayerVisibility when toggling', () => {
		toggleLayer('building');
		
		expect(layerManagerService.toggleLayerVisibility).toHaveBeenCalledWith('building', false);

		toggleLayer('building');
		expect(layerManagerService.toggleLayerVisibility).toHaveBeenCalledWith('building', true);
	});

	it('should set layer visible directly', () => {
		setLayerVisible('road', true);
		expect(get(layerStore).road).toBe(true);

		setLayerVisible('road', false);
		expect(get(layerStore).road).toBe(false);
	});

	it('should call layerManagerService when setting visibility directly', () => {
		setLayerVisible('road', true);
		expect(layerManagerService.toggleLayerVisibility).toHaveBeenCalledWith('road', true);
	});

	it('should update discovered layers', () => {
		setDiscoveredLayers(['building', 'vegetation', 'unknown']);
		const discovered = get(discoveredLayersStore);
		
		expect(discovered).toEqual(['building', 'vegetation', 'unknown']);
	});

	it('should initialize visibility for newly discovered layers', () => {
		setDiscoveredLayers(['building', 'new_building']);
		const store = get(layerStore);
		
		expect(store.building).toBeDefined();
		expect(store.new_building).toBeDefined();
	});
});

