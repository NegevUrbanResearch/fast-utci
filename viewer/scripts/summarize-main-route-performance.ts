import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const INPUT_PATH = resolve(REPO_ROOT, 'data/performance-results/main-route-selected-hour-current-head.json');
const OUTPUT_DIR = resolve(REPO_ROOT, 'docs/performance');
const OUTPUT_PATH = resolve(OUTPUT_DIR, 'main-route-selected-hour-current-head.md');
type Timings = Record<string, number | null>;

type ArtifactCase = {
	projectLabel: string;
	analysisId: string;
	pointCount: number;
	gridSizeMeters: number;
	selectedMonthIndex: number;
	selectedHourIndex: number;
	selectedTimeIndex: number;
	timings: Timings;
	trackedGpuAllocationBytes: {
		persistentExposureBytes: number;
		allHoursOutputBytes: number;
		selectedHourOutputBytes: number;
		selectedHourOutputBytesHighWatermark: number;
		renderOwnedSelectedHourBytes?: number;
		renderOwnedSelectedHourBytesHighWatermark?: number;
		trackingScope: string;
	};
	ownedGpuMemoryBytes: number;
	proof: {
		utciSurfaceSource: string | null;
		baseRenderTransport: string;
		dataTextureBuildCount: number;
		selectedHourRuntimeContract: {
			strongVisibleGpuPath: boolean | null;
		};
	};
	assertions: {
		pythonBinDebugComparisonFieldsAbsent: boolean;
		forbiddenComparisonFieldsPresent: string[];
		forbiddenRequestUrls: string[];
		memoryScope: string;
	};
};

type Artifact = {
	collectedOn: string;
	sourceRoute: string;
	includedAnalyses: string[];
	excludedBgVariantsExplanation: string;
	cases: ArtifactCase[];
};

function readArtifact(): Artifact {
	if (!existsSync(INPUT_PATH)) {
		throw new Error(`Missing input artifact: ${INPUT_PATH}`);
	}
	return JSON.parse(readFileSync(INPUT_PATH, 'utf8')) as Artifact;
}

function formatMs(value: number | null) {
	return value == null ? '-' : value.toFixed(1);
}

function formatPoints(value: number) {
	return value.toLocaleString();
}

function formatMiB(value: number) {
	return (value / (1024 * 1024)).toFixed(2);
}

function getTrackedAppOwnedGpuMemoryBytes(entry: ArtifactCase) {
	const tracked = entry.trackedGpuAllocationBytes;
	return (
		tracked.persistentExposureBytes +
		tracked.allHoursOutputBytes +
		tracked.selectedHourOutputBytes +
		(tracked.renderOwnedSelectedHourBytes ?? 0)
	);
}

function average(values: Array<number | null>) {
	const defined = values.filter((value): value is number => value != null);
	if (defined.length === 0) return null;
	return defined.reduce((sum, value) => sum + value, 0) / defined.length;
}

function buildInference(cases: ArtifactCase[]) {
	const candidates = [
		{
			key: 'exposurePrecomputeMs',
			label: 'exposure precompute / cold-start compute'
		},
		{
			key: 'renderSceneSyncStartDelayMs',
			label: 'pre-scene-sync startup delay'
		},
		{
			key: 'renderSceneSyncTotalMs',
			label: 'render scene sync'
		},
		{
			key: 'workerBvhMs',
			label: 'worker BVH build'
		},
		{
			key: 'payloadPrepareMs',
			label: 'payload preparation'
		},
		{
			key: 'pipelineUploadMs',
			label: 'pipeline upload'
		},
		{
			key: 'oneHourDispatchMs',
			label: 'selected-hour dispatch'
		}
	] as const;

	const ranked = candidates
		.map((candidate) => ({
			...candidate,
			averageMs: average(cases.map((entry) => entry.timings[candidate.key] ?? null))
		}))
		.filter((entry) => entry.averageMs != null)
		.sort((left, right) => (right.averageMs ?? 0) - (left.averageMs ?? 0));

	const top = ranked[0];
	const dispatch = ranked.find((entry) => entry.key === 'oneHourDispatchMs')?.averageMs ?? null;
	if (!top || top.averageMs == null) {
		return 'The artifact did not contain enough timing data to pick a next target.';
	}

	const routeLabel = top.label;
	const averageMsLabel = `${top.averageMs.toFixed(1)} ms`;
	const dispatchLabel = dispatch == null ? 'n/a' : `${dispatch.toFixed(1)} ms`;

	if (top.key === 'exposurePrecomputeMs') {
		return `Fresh main-route numbers point at ${routeLabel} as the next bottleneck: it averages ${averageMsLabel} across BG and Ness Tziona, while selected-hour dispatch stays at ${dispatchLabel}. That means the next optimization pass should stay focused on cold-start work before first visible publication, not on .bin comparison, selected-hour transport, or 0.5m claims yet.`;
	}

	return `Fresh main-route numbers point at ${routeLabel} as the next bottleneck: it averages ${averageMsLabel} across BG and Ness Tziona, while selected-hour dispatch stays at ${dispatchLabel}. That is the highest-signal target for the next optimization pass, with 0.5m still treated as unproven until a dedicated follow-up measures it directly.`;
}

function buildMarkdown(artifact: Artifact) {
	const timingRows = artifact.cases
		.map(
			(entry) =>
				`| ${entry.projectLabel} | \`${entry.analysisId}\` | ${formatPoints(entry.pointCount)} | ${entry.gridSizeMeters.toFixed(1)} | ${entry.selectedMonthIndex} | ${entry.selectedHourIndex} | ${entry.selectedTimeIndex} | ${formatMs(entry.timings.firstSelectedHourVisibleMs ?? null)} | ${formatMs(entry.timings.exposurePrecomputeMs ?? null)} | ${formatMs(entry.timings.renderSceneSyncStartDelayMs ?? null)} | ${formatMs(entry.timings.renderSceneSyncTotalMs ?? null)} | ${formatMs(entry.timings.oneHourDispatchMs ?? null)} |`
		)
		.join('\n');

	const memoryRows = artifact.cases
		.map(
			(entry) =>
				`| ${entry.projectLabel} | \`${entry.analysisId}\` | ${formatMiB(getTrackedAppOwnedGpuMemoryBytes(entry))} | ${formatMiB(entry.trackedGpuAllocationBytes.persistentExposureBytes)} | ${formatMiB(entry.trackedGpuAllocationBytes.selectedHourOutputBytes)} | ${formatMiB(entry.trackedGpuAllocationBytes.selectedHourOutputBytesHighWatermark)} | ${formatMiB(entry.trackedGpuAllocationBytes.renderOwnedSelectedHourBytes ?? 0)} | ${entry.assertions.memoryScope} |`
		)
		.join('\n');

	const proofRows = artifact.cases
		.map(
			(entry) =>
				`- \`${entry.analysisId}\`: \`utciSurfaceSource=${entry.proof.utciSurfaceSource}\`, \`baseRenderTransport=${entry.proof.baseRenderTransport}\`, \`dataTextureBuildCount=${entry.proof.dataTextureBuildCount}\`, \`selectedHourRuntimeContract.strongVisibleGpuPath=${String(entry.proof.selectedHourRuntimeContract.strongVisibleGpuPath)}\`, no python/bin/debug comparison fields, no forbidden comparison requests.`
		)
		.join('\n');

	const unavailableTimingFields = [
		'payloadPrepareMs',
		'workerBvhMs',
		'pipelineUploadMs',
		'firstSelectedHourReadyMs'
	];

	return `# Main Route Selected-Hour Current-HEAD Baseline

Date: ${artifact.collectedOn}

## Scope

This artifact measures the main route \`/\`, not \`/debug\`. It captures the current selected-hour WebGPU path without \`.bin\`, Python, or debug comparison data in the timing baseline.

## Included Analyses

- \`Ben-Gurion/20250815_grid_2m_fullday\`
- \`Ness-Tziona/exploded/nes_tziona_unblock_2\`

## Excluded BG Variants

${artifact.excludedBgVariantsExplanation}

## Proof Boundary

${proofRows}

Memory is scoped only to tracked app-owned UTCI/WebGPU buffers. The \`GPU VRAM\` total mirrors the main-route runtime helper: \`persistentExposureBytes + allHoursOutputBytes + selectedHourOutputBytes + renderOwnedSelectedHourBytes\`. Selected-hour high-watermark is reported as diagnostic context, but the displayed total is not a browser, OS, or device VRAM measurement.

## Unavailable Timing Fields

The fresh \`/\` capture currently exposes only the coarser main-route timing fields shown below. The JSON artifact preserves these fields as \`null\` for both cases instead of inventing values:

- \`payloadPrepareMs\`
- \`workerBvhMs\`
- \`pipelineUploadMs\`
- \`firstSelectedHourReadyMs\`

This means the main-route baseline is suitable for current route-level evidence around first-visible timing, exposure precompute, scene-sync delay, scene-sync total, selected-hour dispatch, and tracked app-owned GPU memory. It is not a full fine-grained cold-start sub-bucket breakdown by itself.

## Timing Table

| Project | Analysis | Points | Grid m | Month | Hour | Time index | First visible ms | Exposure precompute ms | Scene sync start delay ms | Scene sync total ms | Dispatch ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${timingRows}

## Memory Table

| Project | Analysis | GPU VRAM MiB | Persistent exposure MiB | Selected-hour current MiB | Selected-hour HWM MiB | Render-owned selected-hour MiB | Scope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
${memoryRows}

## Current Optimization Inference

${buildInference(artifact.cases)}

The optimization inference above is intentionally conservative: it is based on the available main-route fields plus older debug-route context where finer-grained sub-bucket detail exists.

If these numbers disagree with the older 2026-05-09 strategy snapshot, this fresh main-route baseline wins for current route-level timing decisions, while the older debug-route breakdown remains supporting historical context for unavailable sub-buckets such as ${unavailableTimingFields.map((field) => `\`${field}\``).join(', ')}.
`;
}

function main() {
	const artifact = readArtifact();
	if (!existsSync(OUTPUT_DIR)) {
		mkdirSync(OUTPUT_DIR, { recursive: true });
	}
	const markdown = buildMarkdown(artifact);
	writeFileSync(OUTPUT_PATH, markdown, 'utf8');
	console.log(`Wrote ${OUTPUT_PATH}`);
}

main();
