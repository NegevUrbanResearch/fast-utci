import { describe, it, expect } from 'vitest';
import {
	getDefaultAnalysisId,
	getScenarioCategoriesForProject,
	hasScenarioCategories,
	projects
} from '$lib/config/projects';

describe('projects config', () => {
	it('has Ben-Gurion as default project', () => {
		expect(getDefaultAnalysisId()).toBe('Ben-Gurion/20250815_grid_2m_fullday');
	});

	it('includes Ness-Tziona variants', () => {
		const nt = projects.find((p) => p.id === 'Ness-Tziona');
		expect(nt?.models.length).toBe(1);
	});

	it('exposes scenarios only for projects that define them', () => {
		expect(hasScenarioCategories('Ben-Gurion')).toBe(true);
		expect(getScenarioCategoriesForProject('Ben-Gurion')).toHaveLength(5);

		expect(hasScenarioCategories('Ness-Tziona')).toBe(false);
		expect(getScenarioCategoriesForProject('Ness-Tziona')).toEqual([]);

		expect(hasScenarioCategories('Innovation-District')).toBe(false);
		expect(getScenarioCategoriesForProject('Innovation-District')).toEqual([]);
	});

	it('includes Innovation District as a live WebGPU project', () => {
		const project = projects.find((p) => p.id === 'Innovation-District');
		expect(project?.label).toBe('Innovation District');
		expect(project?.defaultAnalysisId).toBe(
			'Innovation-District/innovation_district_webgpu'
		);
		expect(project?.models).toEqual([
			{
				id: 'base',
				label: 'Base',
				analysisId: 'Innovation-District/innovation_district_webgpu'
			}
		]);
	});
});
