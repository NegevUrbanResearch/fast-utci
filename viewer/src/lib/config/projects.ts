export interface ProjectModel {
	id: string;
	label: string;
	analysisId: string;
}

export interface ScenarioCategory {
	value: string;
	label: string;
	description: string;
}

export interface ProjectConfig {
	id: string;
	label: string;
	defaultAnalysisId: string;
	models: ProjectModel[];
	scenarioCategories?: ScenarioCategory[];
}

export const projects: ProjectConfig[] = [
	{
		id: 'Ben-Gurion',
		label: 'Ben-Gurion',
		defaultAnalysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
		scenarioCategories: [
			{
				value: 'existing_buildings',
				label: 'Existing buildings with added mass',
				description: 'Current buildings made higher'
			},
			{
				value: 'existing_trees',
				label: 'Existing Tree Cover',
				description: 'From no trees up to current canopy'
			},
			{
				value: 'new_high_buildings',
				label: 'New Highrise Buildings',
				description: 'Adds more tall buildings to the site'
			},
			{
				value: 'new_low_buildings',
				label: 'New Lowrise Buildings',
				description: 'Adds more low and mid-rise buildings'
			},
			{
				value: 'new_trees',
				label: 'New Tree Cover',
				description: 'Adds more tree cover'
			}
		],
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

export function getScenarioCategoriesForProject(projectId: string): ScenarioCategory[] {
	return getProjectById(projectId)?.scenarioCategories ?? [];
}

export function hasScenarioCategories(projectId: string): boolean {
	return getScenarioCategoriesForProject(projectId).length > 0;
}
