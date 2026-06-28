import { describe, it, expect } from 'vitest';
import { resolveProjectId, resolveModelPath, resolveAnalysisModelPath } from '$lib/utils/analysisPaths';

describe('analysisPaths', () => {
	it('resolves project id from analysis id', () => {
		expect(resolveProjectId('Ben-Gurion/20250815_grid_2m_fullday')).toBe('Ben-Gurion');
	});

	it('fixes legacy BG model paths', () => {
		const fixed = resolveModelPath(
			'data/3d_models/original_with_layers.glb',
			'Ben-Gurion/20250815_grid_2m_fullday'
		);
		expect(fixed).toBe('data/3d_models/Ben-Gurion/original_with_layers.glb');
	});

	it('keeps already-correct model paths', () => {
		const fixed = resolveModelPath(
			'data/3d_models/Ness-Tziona/nes_tziona_1.gltf',
			'Ness-Tziona/original/20250815_grid_2m_fullday'
		);
		expect(fixed).toBe('data/3d_models/Ness-Tziona/nes_tziona_1.gltf');
	});

	it('does not map legacy Ben-Gurion model paths into another project', () => {
		const fixed = resolveModelPath(
			'data/3d_models/original_with_layers.glb',
			'Ness-Tziona/exploded/nes_tziona_unblock_2'
		);

		expect(fixed).toBe('data/3d_models/Ben-Gurion/original_with_layers.glb');
	});

	it('uses the metadata source analysis id instead of a stale route id', () => {
		const fixed = resolveAnalysisModelPath(
			{
				model_file: 'data/3d_models/original_with_layers.glb',
				source_analysis_id: 'Ben-Gurion/20250815_grid_2m_fullday'
			},
			'Ness-Tziona/exploded/nes_tziona_unblock_2'
		);

		expect(fixed).toBe('data/3d_models/Ben-Gurion/original_with_layers.glb');
	});
});
