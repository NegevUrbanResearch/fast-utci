import type { OnDemandTimings } from '$lib/compute/on-demand/onDemandDiagnostics';
import { formatDuration } from '$lib/performance/mainRoutePerformanceTelemetry';

export interface PythonBinSampleComparisonLike {
	sampleCount?: number;
	maxAbsDiff?: number;
	meanAbsDiff?: number;
}

export interface DebugPerformanceComparisonDiagnosticsLike {
	binComparisonEnabled?: boolean;
	binComparisonValid?: boolean;
	pythonBaselineStatus?: string;
	pythonSelectedHourMeanUtci?: number;
	webgpuSelectedHourMeanUtci?: number;
	pythonDerivedOneHourMs?: number;
	pythonBinSampleComparison?: PythonBinSampleComparisonLike;
	timings?: OnDemandTimings;
}

export interface DebugPerformanceComparisonRow {
	metric: string;
	python: string;
	webgpu: string;
	diff: string;
}

function formatUtci(value: number | undefined): string {
	if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
	return `${value.toFixed(2)} C`;
}

function formatSignedUtciDiff(value: number | undefined): string {
	if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
	const prefix = value > 0 ? '+' : '';
	return `${prefix}${value.toFixed(2)} C`;
}

function formatSignedDurationDiff(valueMs: number | undefined): string {
	if (typeof valueMs !== 'number' || !Number.isFinite(valueMs)) return '-';
	if (valueMs === 0) return formatDuration(0);
	const prefix = valueMs > 0 ? '+' : '-';
	return `${prefix}${formatDuration(Math.abs(valueMs))}`;
}

export function buildDebugPerformanceComparisonRows(
	diagnostics: DebugPerformanceComparisonDiagnosticsLike | null | undefined
): DebugPerformanceComparisonRow[] {
	const webgpuVisibleMs = diagnostics?.timings?.firstSelectedHourVisibleMs;
	const pythonVisibleMs = diagnostics?.pythonDerivedOneHourMs;
	const meanDiff =
		typeof diagnostics?.pythonSelectedHourMeanUtci === 'number' &&
		Number.isFinite(diagnostics.pythonSelectedHourMeanUtci) &&
		typeof diagnostics.webgpuSelectedHourMeanUtci === 'number' &&
		Number.isFinite(diagnostics.webgpuSelectedHourMeanUtci)
			? diagnostics.webgpuSelectedHourMeanUtci - diagnostics.pythonSelectedHourMeanUtci
			: undefined;
	const visibleDiff =
		typeof pythonVisibleMs === 'number' &&
		Number.isFinite(pythonVisibleMs) &&
		typeof webgpuVisibleMs === 'number' &&
		Number.isFinite(webgpuVisibleMs)
			? webgpuVisibleMs - pythonVisibleMs
			: undefined;

	if (!diagnostics?.binComparisonEnabled || !diagnostics.binComparisonValid) {
		return [
			{
				metric: 'Mean UTCI',
				python: 'Unavailable for this selection',
				webgpu: formatUtci(diagnostics?.webgpuSelectedHourMeanUtci),
				diff: '-'
			},
			{
				metric: 'Visible time',
				python: '-',
				webgpu: formatDuration(webgpuVisibleMs ?? null),
				diff: '-'
			}
		];
	}

	return [
		{
			metric: 'Mean UTCI',
			python: formatUtci(diagnostics.pythonSelectedHourMeanUtci),
			webgpu: formatUtci(diagnostics.webgpuSelectedHourMeanUtci),
			diff: formatSignedUtciDiff(meanDiff)
		},
		{
			metric: 'Visible time',
			python: formatDuration(pythonVisibleMs ?? null),
			webgpu: formatDuration(webgpuVisibleMs ?? null),
			diff: formatSignedDurationDiff(visibleDiff)
		}
	];
}
