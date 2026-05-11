import { describe, expect, it } from 'vitest';

import {
	buildProjectSelectionHref,
	getAnalysisSyncAfterMount,
	getMountedAnalysisId,
	getModelReloadState
} from '../../src/routes/main/modelSelection';

describe('main route model selection helpers', () => {
	it('resolves the mounted analysis id from the initial query string', () => {
		expect(
			getMountedAnalysisId('?analysis=Ben-Gurion%2Fscenario-a', 'Ben-Gurion/default')
		).toBe('Ben-Gurion/scenario-a');
	});

	it('keeps the current analysis stable when the URL stays unchanged after mount', () => {
		const result = getAnalysisSyncAfterMount({
			mounted: true,
			currentAnalysisId: 'Ben-Gurion/base',
			pageSearchParams: new URLSearchParams('analysis=Ben-Gurion/base'),
			defaultAnalysisId: 'Ben-Gurion/default'
		});

		expect(result).toEqual({
			analysisId: 'Ben-Gurion/base',
			shouldLoad: false
		});
	});

	it('mutates only the analysis query parameter for project selection navigation', () => {
		const href = buildProjectSelectionHref(
			'https://example.test/viewer?analysis=Ben-Gurion/base&utciRenderDiagnostics=1',
			'BGU/scenario-a'
		);

		expect(href).toBe(
			'/viewer?analysis=BGU%2Fscenario-a&utciRenderDiagnostics=1'
		);
	});

	it('tracks model-file reload bookkeeping when the visible model file changes', () => {
		const result = getModelReloadState({
			currentModelFile: 'data/3d_models/BGU/scenario.glb',
			lastModelFile: 'data/3d_models/BGU/original.glb'
		});

		expect(result).toEqual({
			shouldResetModel: true,
			nextLastModelFile: 'data/3d_models/BGU/scenario.glb'
		});
	});
});
