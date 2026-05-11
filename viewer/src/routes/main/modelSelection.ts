import { getInitialAnalysisId } from '$lib/utils/analysisQuery';

export function getMountedAnalysisId(
	search: string,
	defaultAnalysisId: string,
): string {
	return getInitialAnalysisId(search, defaultAnalysisId);
}

export function getAnalysisSyncAfterMount(params: {
	mounted: boolean;
	currentAnalysisId: string;
	pageSearchParams: URLSearchParams | null | undefined;
	defaultAnalysisId: string;
}): { analysisId: string; shouldLoad: boolean } {
	if (!params.mounted || !params.pageSearchParams) {
		return {
			analysisId: params.currentAnalysisId,
			shouldLoad: false,
		};
	}

	const analysisId = getInitialAnalysisId(
		`?${params.pageSearchParams.toString()}`,
		params.defaultAnalysisId,
	);

	return {
		analysisId,
		shouldLoad: analysisId !== params.currentAnalysisId,
	};
}

export function buildProjectSelectionHref(
	currentHref: string,
	newAnalysisId: string,
): string {
	const url = new URL(currentHref);
	url.searchParams.set('analysis', newAnalysisId);
	return `${url.pathname}?${url.searchParams.toString()}`;
}

export function getModelReloadState(params: {
	currentModelFile: string | null | undefined;
	lastModelFile: string | null;
}): { shouldResetModel: boolean; nextLastModelFile: string | null } {
	const currentModelFile = params.currentModelFile ?? null;

	return {
		shouldResetModel:
			currentModelFile !== null && currentModelFile !== params.lastModelFile,
		nextLastModelFile: currentModelFile,
	};
}
