/**
 * Analysis Store
 * 
 * Svelte store for managing UTCI analysis data state with LRU caching
 */

import { writable, type Writable } from 'svelte/store';
import { base } from '$app/paths';
import type { Analysis } from '$lib/types/analysis';
import { loadAnalysis } from '$lib/services/dataLoader';
import { LRUCache } from '$lib/services/lruCache';

/**
 * Analysis store - holds current analysis data or null
 */
export const analysisStore: Writable<Analysis | null> = writable<Analysis | null>(null);

/**
 * Analysis data cache - stores up to 10 loaded analyses
 * Eviction callback logs but doesn't need to dispose (just JSON + typed arrays)
 */
const analysisCache = new LRUCache<Analysis>({
	maxSize: 10,
	onEvict: (key: string, _value: Analysis) => {
		console.log(`[ANALYSIS CACHE] Evicting analysis: ${key}`);
	}
});

/**
 * Load analysis data and update store
 * Uses LRU cache to avoid reloading previously accessed analyses
 * 
 * @param analysisId - Analysis identifier
 * @param dataDir - Base directory for data files (optional, defaults to base path + "/data/analyses")
 */
export async function loadAnalysisData(
	analysisId: string,
	dataDir?: string
): Promise<void> {
	try {
		// Check cache first
		const cached = analysisCache.get(analysisId);
		if (cached) {
			console.log(`[ANALYSIS CACHE] Cache hit: ${analysisId}`);
			analysisStore.set(cached);
			return;
		}

		// Load from server if not cached
		console.log(`[ANALYSIS CACHE] Cache miss: ${analysisId}, loading...`);
		const analysis = await loadAnalysis(analysisId, dataDir);
		
		// Cache the loaded analysis
		analysisCache.set(analysisId, analysis);
		
		// Update store
		analysisStore.set(analysis);
	} catch (error) {
		console.error('[ERROR] Failed to load analysis:', error);
		throw error;
	}
}

/**
 * Get cache statistics (for debugging/monitoring)
 */
export function getAnalysisCacheStats(): {
	size: number;
	keys: string[];
} {
	return {
		size: analysisCache.size,
		keys: analysisCache.keys()
	};
}

/**
 * Clear the analysis cache
 * Useful for testing or manual cache invalidation
 */
export function clearAnalysisCache(): void {
	analysisCache.clear();
	console.log('[ANALYSIS CACHE] Cache cleared');
}


