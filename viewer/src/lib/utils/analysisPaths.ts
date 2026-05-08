export function resolveProjectId(analysisId?: string | null): string | null {
	if (!analysisId) return null;
	const parts = analysisId.split('/').filter(Boolean);
	return parts.length > 1 ? parts[0] : null;
}

export function resolveModelPath(
	modelFile: string,
	analysisId?: string | null
): string {
	if (!modelFile) return modelFile;

	const basePrefix = 'data/3d_models/';
	if (!modelFile.startsWith(basePrefix)) return modelFile;

	const projectId = resolveProjectId(analysisId);
	if (!projectId) return modelFile;

	const projectPrefix = `${basePrefix}${projectId}/`;
	if (modelFile.startsWith(projectPrefix)) return modelFile;

	const remainder = modelFile.slice(basePrefix.length);
	if (projectId !== 'Ben-Gurion' && isLegacyBenGurionModelPath(remainder)) {
		return `${basePrefix}Ben-Gurion/${remainder}`;
	}

	return `${projectPrefix}${remainder}`;
}

function isLegacyBenGurionModelPath(remainder: string): boolean {
	return remainder === 'original_with_layers.glb' || remainder.startsWith('scenarios/');
}

export function resolveAnalysisModelPath(
	metadata: { model_file: string; source_analysis_id?: string | null },
	fallbackAnalysisId?: string | null
): string {
	return resolveModelPath(
		metadata.model_file,
		metadata.source_analysis_id ?? fallbackAnalysisId
	);
}
