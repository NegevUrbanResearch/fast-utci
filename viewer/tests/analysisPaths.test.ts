import { describe, it, expect } from 'vitest';
import { resolveProjectId, resolveModelPath } from '$lib/utils/analysisPaths';

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
});
