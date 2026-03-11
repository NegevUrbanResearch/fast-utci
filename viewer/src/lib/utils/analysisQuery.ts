export function getInitialAnalysisId(
	search: string,
	defaultId: string
): string {
	if (!search) return defaultId;
	const params = new URLSearchParams(search);
	return params.get('analysis') || defaultId;
}
