import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { render, cleanup } from '@testing-library/svelte/svelte5';
import { tick } from 'svelte';
import { Group } from 'three';
import { get } from 'svelte/store';
import Model from '$lib/components/scene/Model.svelte';
import { analysisStore, clearAnalysisCache, loadAnalysisData } from '$lib/stores/analysisStore';
import { loadAnalysis } from '$lib/services/dataLoader';
import { applyLayerMaterials } from '$lib/services/modelLoaderService';
import { viewerStore, setError, setLoading } from '$lib/stores/viewerStore';
import { modelLoadHandlers } from '../mocks/modelLoadHarness';
import type { ThrelteGltf } from '@threlte/extras';
import type { Analysis } from '$lib/types/analysis';

vi.mock('@threlte/extras', async () => ({ GLTF: (await import('../mocks/ModelGltf.svelte')).default }));

vi.mock('@threlte/core', () => ({ T: {} }));

vi.mock('$lib/services/modelLoaderService', () => ({ applyLayerMaterials: vi.fn() }));

vi.mock('$lib/services/modelCacheService', () => ({
	getCachedModel: () => undefined,
	hasModelInCache: () => false,
	cacheModel: vi.fn()
}));

vi.mock('$lib/services/layerManagerService', () => ({
	initializeLayerManager: vi.fn(),
	resetLayerManager: vi.fn(),
	applyLayerVisibilityState: vi.fn(),
	getDiscoveredLayers: () => ['building']
}));

vi.mock('$lib/config/viewerConfig', () => ({ isNormalizationEnabled: () => false }));

vi.mock('$lib/services/dataLoader', () => ({
	loadAnalysis: vi.fn(),
	loadAnalysisMetadataOnly: vi.fn()
}));

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: Error) => void;
	const promise = new Promise<T>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return {
		promise,
		resolve,
		reject
	};
}

beforeEach(() => {
	vi.resetAllMocks();
	clearAnalysisCache();
	analysisStore.set(null);
	setError(null);
	setLoading(false);
	modelLoadHandlers.length = 0;
});

afterEach(cleanup);

for (const failure of [false, true]) {
	it(`rejects A ${failure ? 'error' : 'publication'} while B metadata is pending`, async () => {
		const materials = deferred<Group>();
		const metadata = deferred<Analysis>();
		vi.mocked(applyLayerMaterials).mockReturnValueOnce(materials.promise);
		let selection = 1;
		const loaded = vi.fn(() => setLoading(false));
		const layers = vi.fn();
		const component = render(Model, {
			props: {
				modelPath: 'a.glb',
				isRequestCurrent: () => selection === 1
			}
		});
		component.component.$on('modelLoaded', loaded);
		component.component.$on('layersDiscovered', layers);
		await tick();
		modelLoadHandlers[0].onload({ scene: new Group() } as ThrelteGltf);
		selection = 2;
		setLoading(true);
		vi.mocked(loadAnalysis).mockReturnValueOnce(metadata.promise);
		const next = loadAnalysisData('b');
		if (failure)
			materials.reject(new Error('stale A'));
		else
			materials.resolve(new Group());
		await tick();
		await tick();
		expect(loaded).not.toHaveBeenCalled();
		expect(layers).not.toHaveBeenCalled();
		expect(get(viewerStore).loading).toBe(true);
		expect(get(viewerStore).error).toBeNull();
		metadata.resolve({ metadata: { model_file: 'b.glb' } } as Analysis);
		await next;
	});
}

it('invalidates pending material processing when the mounted model is destroyed', async () => {
	const materials = deferred<Group>();
	vi.mocked(applyLayerMaterials).mockReturnValueOnce(materials.promise);
	const loaded = vi.fn();
	const component = render(Model, { props: { modelPath: 'a.glb' } });
	component.component.$on('modelLoaded', loaded);
	await tick();
	modelLoadHandlers[0].onload({ scene: new Group() } as ThrelteGltf);
	component.unmount();
	materials.resolve(new Group());
	await tick();
	await tick();
	expect(loaded).not.toHaveBeenCalled();
	expect(get(viewerStore).error).toBeNull();
});

it('reports the latest mounted model material failure through viewer state', async () => {
	const materials = deferred<Group>();
	vi.mocked(applyLayerMaterials).mockReturnValueOnce(materials.promise);
	render(Model, { props: { modelPath: 'a.glb' } });
	await tick();
	modelLoadHandlers[0].onload({ scene: new Group() } as ThrelteGltf);
	materials.reject(new Error('current material failure'));
	await tick();
	await tick();
	expect(get(viewerStore).error).toBe('current material failure');
	expect(get(viewerStore).loading).toBe(false);
});
