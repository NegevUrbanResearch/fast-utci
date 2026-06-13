export interface ProjectModel {
	id: string;
	label: string;
	analysisId: string;
}

export interface ProjectConfig {
	id: string;
	label: string;
	defaultAnalysisId: string;
	models: ProjectModel[];
}

export const projects: ProjectConfig[] = [
	{
		id: 'Ben-Gurion',
		label: 'Ben-Gurion',
		defaultAnalysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		models: [
			{
				id: 'base',
				label: 'Base',
				analysisId: 'Ben-Gurion/20250815_grid_2m_fullday'
			}
		]
	},
	{
		id: 'Ness-Tziona',
		label: 'Ness-Tziona',
		defaultAnalysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2',
		models: [
			{
				id: 'exploded',
				label: 'Exploded',
				analysisId: 'Ness-Tziona/exploded/nes_tziona_unblock_2'
			}
		]
	},
	{
		id: 'Innovation-District',
		label: 'Innovation District',
		defaultAnalysisId: 'Innovation-District/innovation_district_webgpu',
		models: [
			{
				id: 'base',
				label: 'Base',
				analysisId: 'Innovation-District/innovation_district_webgpu'
			}
		]
	}
];

export function getDefaultAnalysisId(): string {
	return projects[0]?.defaultAnalysisId ?? '';
}

export function getProjectById(id: string): ProjectConfig | undefined {
	return projects.find((project) => project.id === id);
}
