import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const artifactPath = resolve(
	repoRoot,
	'data/performance-results/main-route-transition-scrub-diagnostics.json'
);

type Timeline = Record<string, unknown>;
type Sample = {
	actionKind?: string;
	actionLabel?: string;
	colorMode?: string | null;
	proof?: {
		rendererBackend?: string;
		utciSurfaceSource?: string;
		baseRenderTransport?: string;
		baseSameDeviceForComputeAndRender?: boolean;
		dataTextureBuildCount?: number | null;
		selectedHourRuntimeContract?: {
			visibleSelectedHourReadbackCount?: number | null;
			strongVisibleGpuPath?: boolean | null;
		};
	};
	forbiddenComparisonFieldsPresent?: string[];
	renderPublication?: {
		timeline?: Timeline | null;
	};
};
type Artifact = { cases?: Array<{ caseId?: string; samples?: Sample[] }> };

function numberField(timeline: Timeline, key: string): number | null {
	const value = timeline[key];
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringField(timeline: Timeline, key: string): string | null {
	const value = timeline[key];
	return typeof value === 'string' ? value : null;
}

const artifact = JSON.parse(readFileSync(artifactPath, 'utf8')) as Artifact;
const expectedUncachedMonthProofCount = Math.max(0, (artifact.cases?.length ?? 0) * 2);
const expectedPerHourProofCount = Math.max(0, (artifact.cases?.length ?? 0) * 2);
let badProof = 0;
let forbidden = 0;
let missingSelectedDayCompact = 0;
let missingPerHourCompact = 0;
let uncachedMonthProofCount = 0;
let perHourProofCount = 0;
let samples = 0;

for (const c of artifact.cases ?? []) {
	for (const s of c.samples ?? []) {
		samples += 1;
		const proof = s.proof ?? {};
		const contract = proof.selectedHourRuntimeContract ?? {};
		if (
			proof.rendererBackend !== 'webgpu' ||
			proof.utciSurfaceSource !== 'compute-buffer-selected-hour' ||
			proof.baseRenderTransport !== 'compute-buffer-selected-hour' ||
			proof.baseSameDeviceForComputeAndRender !== true ||
			proof.dataTextureBuildCount !== 0 ||
			contract.visibleSelectedHourReadbackCount !== 0 ||
			contract.strongVisibleGpuPath !== true
		) {
			badProof += 1;
		}
		forbidden += s.forbiddenComparisonFieldsPresent?.length ?? 0;
		const timeline = s.renderPublication?.timeline ?? null;
		if (!timeline) continue;
		if (s.actionKind === 'month-change' && timeline.sessionSelectedDayRangeCacheHit === false) {
			uncachedMonthProofCount += 1;
			if (
				stringField(timeline, 'sessionSelectedDayRangeResolutionPath') !==
					'compact-gpu-summary' ||
				numberField(timeline, 'sessionSelectedDayRangeReadbackCount') !== 0 ||
				numberField(timeline, 'sessionSelectedDayRangeSummaryReadbackCount') !== 23 ||
				numberField(timeline, 'sessionSelectedDayRangeSummaryReadbackBytes') !== 23 * 16 ||
				numberField(timeline, 'sessionSelectedDayRangeFullReadbackAvoidedCount') !== 23
			) {
				missingSelectedDayCompact += 1;
			}
		}
		if (s.colorMode === 'discrete') {
			perHourProofCount += 1;
			if (
				stringField(timeline, 'sessionSelectedHourRangeResolutionPath') !==
					'compact-gpu-summary' ||
				numberField(timeline, 'sessionSelectedHourRangeReadbackCount') !== 0 ||
				numberField(timeline, 'sessionSelectedHourRangeCpuScanCount') !== 0 ||
				numberField(timeline, 'sessionSelectedHourRangeSummaryReadbackCount') !== 1 ||
				numberField(timeline, 'sessionSelectedHourRangeSummaryReadbackBytes') !== 16 ||
				numberField(timeline, 'sessionSelectedHourRangeFullReadbackAvoidedCount') !== 1
			) {
				missingPerHourCompact += 1;
			}
		}
	}
}

const result = {
	cases: artifact.cases?.length ?? 0,
	samples,
	badProof,
	forbidden,
	missingSelectedDayCompact,
	missingPerHourCompact,
	uncachedMonthProofCount,
	perHourProofCount,
	expectedUncachedMonthProofCount,
	expectedPerHourProofCount
};
console.log(JSON.stringify(result, null, 2));

if (
	result.cases < 4 ||
	badProof !== 0 ||
	forbidden !== 0 ||
	missingSelectedDayCompact !== 0 ||
	missingPerHourCompact !== 0 ||
	uncachedMonthProofCount < expectedUncachedMonthProofCount ||
	perHourProofCount < expectedPerHourProofCount
) {
	process.exitCode = 1;
}
