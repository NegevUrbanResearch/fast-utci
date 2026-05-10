import { resolveProjectId } from '$lib/utils/analysisPaths';

const PROJECT_EPW_PATHS: Record<string, string> = {
	'Ben-Gurion':
		'/data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw',
	'Ness-Tziona':
		'/data/weather/ISR_TA_Tel.Aviv-Bet.Dagan.401790_TMYx/ISR_TA_Tel.Aviv-Bet.Dagan.401790_TMYx.epw'
};

export function getEpwUrlForProject(params: {
	projectId: string;
	dataBasePath: string;
}): string {
	const { projectId, dataBasePath } = params;
	const relativePath = PROJECT_EPW_PATHS[projectId];
	if (!relativePath) {
		throw new Error(`No EPW weather mapping is configured for project "${projectId}".`);
	}

	return `${dataBasePath}${relativePath}`;
}

export function getEpwUrlForAnalysis(params: {
	analysisId?: string | null;
	dataBasePath: string;
	fallbackProjectId?: string | null;
}): string {
	const projectId = resolveProjectId(params.analysisId) ?? params.fallbackProjectId;
	if (!projectId) {
		throw new Error('Unable to resolve a project id for EPW weather selection.');
	}
	return getEpwUrlForProject({
		projectId,
		dataBasePath: params.dataBasePath
	});
}
