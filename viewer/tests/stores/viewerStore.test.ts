import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { viewerStore, setCurrentHour, setColorMode, setUtciVisible, setLoading, setError } from '$lib/stores/viewerStore';
import type { ColorMode } from '$lib/types/viewer';

describe('viewerStore', () => {
	beforeEach(() => {
		// Reset store before each test
		viewerStore.set({
			currentHour: 0,
			currentMonth: 7,
			colorMode: 'normalized',
			metricType: 'utci',
			utciVisible: true,
			analysisId: null,
			loading: false,
			error: null,
			theme: 'dark'
		});
	});

	it('should initialize with default values', () => {
		const store = get(viewerStore);
		expect(store.currentHour).toBe(0);
		expect(store.colorMode).toBe('normalized');
		expect(store.utciVisible).toBe(true);
		expect(store.analysisId).toBeNull();
		expect(store.loading).toBe(false);
		expect(store.error).toBeNull();
	});

	it('should set current hour', () => {
		setCurrentHour(12);
		const store = get(viewerStore);
		expect(store.currentHour).toBe(12);
	});

	it('should set color mode', () => {
		setColorMode('discrete');
		const store = get(viewerStore);
		expect(store.colorMode).toBe('discrete');

		setColorMode('normalized');
		expect(get(viewerStore).colorMode).toBe('normalized');
	});

	it('should set UTCI visible', () => {
		setUtciVisible(false);
		const store = get(viewerStore);
		expect(store.utciVisible).toBe(false);

		setUtciVisible(true);
		expect(get(viewerStore).utciVisible).toBe(true);
	});

	it('should set loading state', () => {
		setLoading(true);
		const store = get(viewerStore);
		expect(store.loading).toBe(true);

		setLoading(false);
		expect(get(viewerStore).loading).toBe(false);
	});

	it('should set error message', () => {
		setError('Test error');
		const store = get(viewerStore);
		expect(store.error).toBe('Test error');

		setError(null);
		expect(get(viewerStore).error).toBeNull();
	});
});


