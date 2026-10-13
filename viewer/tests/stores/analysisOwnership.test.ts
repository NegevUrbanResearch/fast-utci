import { describe, it, expect, beforeEach, vi } from 'vitest';
import { get } from 'svelte/store';
import { analysisStore, clearAnalysisCache, getAnalysisCacheStats, loadAnalysisData } from '$lib/stores/analysisStore';
import { loadAnalysis, loadAnalysisMetadataOnly } from '$lib/services/dataLoader';
import type { Analysis } from '$lib/types/analysis';
vi.mock('$lib/services/dataLoader', () => ({ loadAnalysis: vi.fn(), loadAnalysisMetadataOnly: vi.fn() }));
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (error: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
const a = { metadata: { model_file: 'a' } } as Analysis;
const b = { metadata: { model_file: 'b' } } as Analysis;
beforeEach(() => { clearAnalysisCache(); analysisStore.set(null); vi.resetAllMocks(); });
describe('analysis request ownership', () => {
	it('keeps B when A completes later and caches both', async () => { const pending = deferred<Analysis>(); vi.mocked(loadAnalysis).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(b); const first = loadAnalysisData('a'); await loadAnalysisData('b'); pending.resolve(a); await first; expect(get(analysisStore)).toBe(b); expect(getAnalysisCacheStats().size).toBe(2); });
	it('gives a cached B ownership over pending A', async () => { vi.mocked(loadAnalysis).mockResolvedValueOnce(b); await loadAnalysisData('b'); const pending = deferred<Analysis>(); vi.mocked(loadAnalysis).mockReturnValueOnce(pending.promise); const first = loadAnalysisData('a'); await loadAnalysisData('b'); pending.resolve(a); await first; expect(get(analysisStore)).toBe(b); });
	it('settles a stale rejection without sending it to the route error handler', async () => { const pending = deferred<Analysis>(); vi.mocked(loadAnalysis).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(b); const first = loadAnalysisData('a'); await loadAnalysisData('b'); pending.reject(new Error('stale')); await expect(first).resolves.toBeUndefined(); expect(get(analysisStore)).toBe(b); });
	it('propagates current failures', async () => { vi.mocked(loadAnalysis).mockRejectedValueOnce(new Error('current')); await expect(loadAnalysisData('a')).rejects.toThrow('current'); });
	it('distinguishes data roots and metadata mode', async () => { vi.mocked(loadAnalysis).mockResolvedValueOnce(a).mockResolvedValueOnce(b); vi.mocked(loadAnalysisMetadataOnly).mockResolvedValueOnce(a); await loadAnalysisData('same', '/one'); await loadAnalysisData('same', '/two'); expect(get(analysisStore)).toBe(b); await loadAnalysisData('same', '/two', { metadataOnly: true }); expect(loadAnalysisMetadataOnly).toHaveBeenCalledOnce(); expect(getAnalysisCacheStats().size).toBe(3); });
	it('clearing invalidates pending publication and cache insertion', async () => { const pending = deferred<Analysis>(); vi.mocked(loadAnalysis).mockReturnValueOnce(pending.promise); const first = loadAnalysisData('a'); clearAnalysisCache(); pending.resolve(a); await first; expect(get(analysisStore)).toBeNull(); expect(getAnalysisCacheStats().size).toBe(0); });
});
