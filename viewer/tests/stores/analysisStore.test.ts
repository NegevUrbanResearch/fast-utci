import { describe, it, expect, beforeEach, vi } from 'vitest';
import { get } from 'svelte/store';
import { 
	analysisStore, 
	loadAnalysisData, 
	getAnalysisCacheStats, 
	clearAnalysisCache 
} from '$lib/stores/analysisStore';
import type { Analysis } from '$lib/types/analysis';
import * as dataLoader from '$lib/services/dataLoader';

// Mock the dataLoader
vi.mock('$lib/services/dataLoader', () => ({
	loadAnalysis: vi.fn()
}));

describe('analysisStore', () => {
	beforeEach(() => {
		// Reset store and cache before each test
		analysisStore.set(null);
		clearAnalysisCache();
		vi.clearAllMocks();
	});

	it('should initialize with null', () => {
		const store = get(analysisStore);
		expect(store).toBeNull();
	});

	it('should set analysis data', () => {
		const mockAnalysis: Analysis = {
			metadata: {
				analysis_type: 'single_hour',
				num_positions: 100,
				hours: ['12:00'],
				utci_range: { min: 10, max: 30 },
				grid_size: 2.0,
				coordinate_system: 'xy_ground',
				model_file: 'test.glb'
			},
			data: {
				numPositions: 100,
				numHours: 1,
				positions: new Float32Array(300),
				utciValues: new Float32Array(100)
			}
		};

		analysisStore.set(mockAnalysis);
		const store = get(analysisStore);
		expect(store).toEqual(mockAnalysis);
	});

	it('should update analysis data', () => {
		const mockAnalysis1: Analysis = {
			metadata: {
				analysis_type: 'single_hour',
				num_positions: 100,
				hours: ['12:00'],
				utci_range: { min: 10, max: 30 },
				grid_size: 2.0,
				coordinate_system: 'xy_ground',
				model_file: 'test1.glb'
			},
			data: {
				numPositions: 100,
				numHours: 1,
				positions: new Float32Array(300),
				utciValues: new Float32Array(100)
			}
		};

		const mockAnalysis2: Analysis = {
			metadata: {
				analysis_type: 'full_day',
				num_positions: 200,
				hours: ['00:00', '01:00'],
				utci_range: { min: 5, max: 35 },
				grid_size: 2.0,
				coordinate_system: 'xy_ground',
				model_file: 'test2.glb'
			},
			data: {
				numPositions: 200,
				numHours: 2,
				positions: new Float32Array(600),
				utciByHour: [new Float32Array(200), new Float32Array(200)]
			}
		};

		analysisStore.set(mockAnalysis1);
		expect(get(analysisStore)?.metadata.model_file).toBe('test1.glb');

		analysisStore.set(mockAnalysis2);
		expect(get(analysisStore)?.metadata.model_file).toBe('test2.glb');
	});

	describe('Caching', () => {
		const mockAnalysis: Analysis = {
			metadata: {
				analysis_type: 'single_hour',
				num_positions: 100,
				hours: ['12:00'],
				utci_range: { min: 10, max: 30 },
				grid_size: 2.0,
				coordinate_system: 'xy_ground',
				model_file: 'test.glb'
			},
			data: {
				numPositions: 100,
				numHours: 1,
				positions: new Float32Array(300),
				utciValues: new Float32Array(100)
			}
		};

		it('should cache loaded analysis data', async () => {
			vi.mocked(dataLoader.loadAnalysis).mockResolvedValue(mockAnalysis);

			await loadAnalysisData('test-id');

			const stats = getAnalysisCacheStats();
			expect(stats.size).toBe(1);
			expect(stats.keys).toContain('test-id');
		});

		it('should use cached data on subsequent loads', async () => {
			vi.mocked(dataLoader.loadAnalysis).mockResolvedValue(mockAnalysis);

			// First load - should hit the loader
			await loadAnalysisData('test-id');
			expect(dataLoader.loadAnalysis).toHaveBeenCalledTimes(1);

			// Second load - should use cache
			await loadAnalysisData('test-id');
			expect(dataLoader.loadAnalysis).toHaveBeenCalledTimes(1); // Still 1, not called again

			const stats = getAnalysisCacheStats();
			expect(stats.size).toBe(1);
		});

		it('should handle multiple different analyses', async () => {
			const mockAnalysis2: Analysis = {
				...mockAnalysis,
				metadata: { ...mockAnalysis.metadata, model_file: 'test2.glb' }
			};

			vi.mocked(dataLoader.loadAnalysis)
				.mockResolvedValueOnce(mockAnalysis)
				.mockResolvedValueOnce(mockAnalysis2);

			await loadAnalysisData('test-id-1');
			await loadAnalysisData('test-id-2');

			const stats = getAnalysisCacheStats();
			expect(stats.size).toBe(2);
			expect(stats.keys).toContain('test-id-1');
			expect(stats.keys).toContain('test-id-2');
		});

		it('should update LRU order when accessing cached data', async () => {
			const mockAnalysis2: Analysis = {
				...mockAnalysis,
				metadata: { ...mockAnalysis.metadata, model_file: 'test2.glb' }
			};

			vi.mocked(dataLoader.loadAnalysis)
				.mockResolvedValueOnce(mockAnalysis)
				.mockResolvedValueOnce(mockAnalysis2);

			// Load two analyses
			await loadAnalysisData('test-id-1');
			await loadAnalysisData('test-id-2');

			// Access first one again (makes it most recently used)
			await loadAnalysisData('test-id-1');

			const stats = getAnalysisCacheStats();
			// LRU order should be: test-id-2, test-id-1
			expect(stats.keys).toEqual(['test-id-2', 'test-id-1']);
		});

		it('should clear cache when requested', async () => {
			vi.mocked(dataLoader.loadAnalysis).mockResolvedValue(mockAnalysis);

			await loadAnalysisData('test-id');
			expect(getAnalysisCacheStats().size).toBe(1);

			clearAnalysisCache();
			expect(getAnalysisCacheStats().size).toBe(0);
		});

		it('should evict least recently used when cache is full', async () => {
			// Create 11 mock analyses (cache max is 10)
			const analyses = Array.from({ length: 11 }, (_, i) => ({
				...mockAnalysis,
				metadata: { ...mockAnalysis.metadata, model_file: `test${i}.glb` }
			}));

			// Mock loadAnalysis to return each analysis in sequence
			for (let i = 0; i < 11; i++) {
				vi.mocked(dataLoader.loadAnalysis).mockResolvedValueOnce(analyses[i]);
			}

			// Load 10 analyses
			for (let i = 0; i < 10; i++) {
				await loadAnalysisData(`test-id-${i}`);
			}

			expect(getAnalysisCacheStats().size).toBe(10);

			// Load 11th analysis - should evict the first one
			await loadAnalysisData('test-id-10');

			const stats = getAnalysisCacheStats();
			expect(stats.size).toBe(10);
			expect(stats.keys).not.toContain('test-id-0');
			expect(stats.keys).toContain('test-id-10');
		});

		it('should handle load errors gracefully', async () => {
			vi.mocked(dataLoader.loadAnalysis).mockRejectedValue(new Error('Load failed'));

			await expect(loadAnalysisData('bad-id')).rejects.toThrow('Load failed');

			// Failed load should not be cached
			const stats = getAnalysisCacheStats();
			expect(stats.size).toBe(0);
		});
	});
});


