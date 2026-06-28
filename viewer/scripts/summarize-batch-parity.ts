import { resolve, join } from 'node:path';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';

const execAsync = promisify(exec);
const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const resultsDir = resolve(REPO_ROOT, 'data/batch-parity-results');
const legacyReportPath = resolve(REPO_ROOT, 'data/batch-parity-results/parity_performance_report.md');
const reportFilename =
	process.env.BATCH_PARITY_REPORT_FILENAME ?? 'parity_performance_report_run_all_vs_on_demand.md';
const reportPath = resolve(REPO_ROOT, 'data/batch-parity-results', reportFilename);

const CONCURRENCY = Math.max(1, Math.min(os.cpus().length - 1, 8));
const GRASSHOPPER_ONE_HOUR_BASELINE_S = 15 * 60;
const REPRESENTATIVE_FULL_YEAR_HOURS = 12 * 24;

function numberOrNull(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function median(values: number[]): number | null {
	if (values.length === 0) return null;
	const sorted = [...values].sort((a, b) => a - b);
	const middle = Math.floor(sorted.length / 2);
	if (sorted.length % 2 === 0) {
		return (sorted[middle - 1] + sorted[middle]) / 2;
	}
	return sorted[middle];
}

function formatFixed(value: number | null | undefined, digits = 2): string {
	return value == null ? '-' : value.toFixed(digits);
}

function formatInt(value: number | null | undefined): string {
	return value == null ? '-' : Math.round(value).toLocaleString();
}

function formatDurationHuman(seconds: number | null | undefined): string {
	if (seconds == null) return '-';
	if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
	if (seconds >= 60) return `${(seconds / 60).toFixed(1)} min`;
	if (seconds >= 1) return `${seconds.toFixed(2)} s`;
	return `${(seconds * 1000).toFixed(1)} ms`;
}

function formatSpeedup(baselineSeconds: number | null | undefined, currentSeconds: number | null | undefined): string {
	if (baselineSeconds == null || currentSeconds == null || currentSeconds <= 0) return '-';
	return `${(baselineSeconds / currentSeconds).toFixed(1)}x`;
}

function formatMiB(bytes: number | null | undefined): string {
	if (bytes == null) return '-';
	return (bytes / (1024 * 1024)).toFixed(2);
}

function formatKiB(bytes: number | null | undefined): string {
	if (bytes == null) return '-';
	return (bytes / 1024).toFixed(1);
}

function calculateDurations(log: any[] | undefined) {
	const getStep = (phase: string) => log?.find((l: any) => l.phase === phase)?.timestamp;

	const tStart = getStep('preflight');
	const tEpw = getStep('epw');
	const tInit = getStep('pipelineInit');
	const tWorker = getStep('worker');
	const tRun = getStep('runAll');
	const tReadback = getStep('readback');
	const tDone = getStep('done');

	const dur = {
		weather: tInit && tEpw ? tInit - tEpw : 0,
		init: tWorker && tInit ? tWorker - tInit : 0,
		bvh: tRun && tWorker ? tRun - tWorker : 0,
		compute: tReadback && tRun ? tReadback - tRun : 0,
		readback: tDone && tReadback ? tDone - tReadback : 0,
		total: tDone && tStart ? tDone - tStart : 0
	};

	const total = Math.max(0.001, dur.total);
	const pct = (value: number) => ((value / total) * 100).toFixed(1) + '%';

	return {
		weather: { s: dur.weather / 1000, pct: pct(dur.weather) },
		init: { s: dur.init / 1000, pct: pct(dur.init) },
		bvh: { s: dur.bvh / 1000, pct: pct(dur.bvh) },
		compute: { s: dur.compute / 1000, pct: pct(dur.compute) },
		readback: { s: dur.readback / 1000, pct: pct(dur.readback) },
		total: { s: dur.total / 1000 }
	};
}

function summarizeOnDemand(timing: any) {
	const onDemand = timing.webgpu_on_demand ?? {};
	const diagnostics = onDemand.diagnostics ?? {};
	const tracked = diagnostics.trackedGpuAllocationBytes ?? {};
	const timings = diagnostics.timings ?? {};
	const representativeHours =
		numberOrNull(timing.representativeFullYearHours) ?? REPRESENTATIVE_FULL_YEAR_HOURS;
	const readyWallS = numberOrNull(onDemand.ready_s);
	const exposurePrecomputeMs = numberOrNull(timings.exposurePrecomputeMs);
	const oneHourDispatchMs = numberOrNull(timings.oneHourDispatchMs);
	const firstHourComputeS =
		exposurePrecomputeMs != null && oneHourDispatchMs != null
			? (exposurePrecomputeMs + oneHourDispatchMs) / 1000
			: null;
	const fullYearEstimateS =
		exposurePrecomputeMs != null && oneHourDispatchMs != null
			? (exposurePrecomputeMs + oneHourDispatchMs * representativeHours) / 1000
			: null;
	const pointCount =
		numberOrNull(diagnostics.pointCount) ??
		numberOrNull(timing.preflight?.numPoints) ??
		numberOrNull(timing.preflight?.estimatedGridPoints);
	const zeroAllHoursAlloc =
		diagnostics && typeof diagnostics === 'object'
			? diagnostics.allHoursUtciBytesAllocated === 0 &&
				diagnostics.allHoursMrtBytesAllocated === 0 &&
				(tracked.allHoursOutputBytes ?? 0) === 0
			: null;

	return {
		state: typeof onDemand.state === 'string' ? onDemand.state : 'missing',
		statusText: typeof onDemand.statusText === 'string' ? onDemand.statusText : null,
		error: typeof onDemand.error === 'string' ? onDemand.error : null,
		readyWallS,
		exposurePrecomputeMs,
		oneHourDispatchMs,
		firstHourComputeS,
		fullYearEstimateS,
		pointCount,
		oneHourOutputBytes: numberOrNull(diagnostics.oneHourOutputBytes),
		allHoursUtciBytesAllocated: numberOrNull(diagnostics.allHoursUtciBytesAllocated),
		allHoursMrtBytesAllocated: numberOrNull(diagnostics.allHoursMrtBytesAllocated),
		persistentExposureBytes: numberOrNull(tracked.persistentExposureBytes),
		selectedHourOutputBytes: numberOrNull(tracked.selectedHourOutputBytes),
		selectedHourOutputBytesHighWatermark: numberOrNull(
			tracked.selectedHourOutputBytesHighWatermark
		),
		allHoursOutputBytes: numberOrNull(tracked.allHoursOutputBytes),
		renderTransport:
			typeof diagnostics.renderTransport === 'string' ? diagnostics.renderTransport : null,
		path: typeof diagnostics.path === 'string' ? diagnostics.path : null,
		zeroAllHoursAlloc,
		diagnostics
	};
}

async function main() {
	if (!existsSync(resultsDir)) {
		console.error('Results directory not found');
		return;
	}

	const files = readdirSync(resultsDir).filter((file) => file.endsWith('_timing.json'));
	console.log(`Found ${files.length} timing files. Processing with concurrency ${CONCURRENCY}...`);

	const results: any[] = [];
	const queue = [...files];
	const activePromises: Promise<void>[] = [];

	const processFile = async (file: string) => {
		const timing = JSON.parse(readFileSync(join(resultsDir, file), 'utf8'));
		const analysisId = timing.analysisId;
		console.log(`Analyzing ${analysisId}...`);

		let parityStatus = { solar: '-', sky: '-', mrt: '-', utci: '-' };
		try {
			const compareCmd = `npx tsx scripts/compare-parity.ts --base-path data/analyses/${analysisId} --mode stats`;
			let output = '';
			try {
				const { stdout } = await execAsync(compareCmd, { cwd: resolve(REPO_ROOT, 'viewer') });
				output = stdout;
			} catch (error: any) {
				output = error.stdout || '';
			}

			const parseStatus = (layer: string) => {
				if (output.includes(`${layer}: PASS`)) return 'PASS';
				if (output.includes(`${layer}: FAIL (no ref or error)`)) return '-';
				if (output.includes(`${layer}: FAIL`)) return 'FAIL';
				return '-';
			};

			parityStatus = {
				solar: parseStatus('solar'),
				sky: parseStatus('sky'),
				mrt: parseStatus('mrt'),
				utci: parseStatus('utci')
			};
		} catch (error) {
			console.warn(`Comparison failed for ${analysisId}`);
		}

		const durParityDay = calculateDurations(timing.webgpu_1m?.log);
		const durFullYear = calculateDurations(timing.webgpu_12m?.log);
		const onDemand = summarizeOnDemand(timing);

		const numPoints =
			numberOrNull(timing.preflight?.numPoints) ??
			numberOrNull(timing.preflight?.estimatedGridPoints) ??
			onDemand.pointCount ??
			0;
		const legacyStoreAll = {
			solar: (numPoints * 24 * 12 / 8) / (1024 * 1024),
			sky: (numPoints * 145 * 4) / (1024 * 1024),
			utci: (numPoints * 24 * 12 * 4) / (1024 * 1024),
			total: 0
		};
		legacyStoreAll.total = legacyStoreAll.solar + legacyStoreAll.sky + legacyStoreAll.utci;

		results.push({
			timing,
			parityStatus,
			durParityDay,
			durFullYear,
			legacyStoreAll,
			onDemand,
			numPoints
		});
	};

	const startWorker = async () => {
		while (queue.length > 0) {
			const file = queue.shift();
			if (file) await processFile(file);
		}
	};

	for (let i = 0; i < CONCURRENCY; i += 1) {
		activePromises.push(startWorker());
	}
	await Promise.all(activePromises);

	results.sort((left, right) => left.timing.analysisId.localeCompare(right.timing.analysisId));

	const grasshopperTwelveMonthBaselineS =
		GRASSHOPPER_ONE_HOUR_BASELINE_S * REPRESENTATIVE_FULL_YEAR_HOURS;
	const onDemandReadyValues = results
		.map((result) => result.onDemand.readyWallS)
		.filter((value): value is number => value != null);
	const onDemandFullYearEstimateValues = results
		.map((result) => result.onDemand.fullYearEstimateS)
		.filter((value): value is number => value != null);
	const onDemandPointValues = results
		.map((result) => result.onDemand.pointCount)
		.filter((value): value is number => value != null);
	const medianProfile = {
		label: `Median (${onDemandReadyValues.length} analyses)`,
		points: median(onDemandPointValues),
		oneHourNowS: median(onDemandReadyValues),
		twelveMonthNowS: median(onDemandFullYearEstimateValues),
		note: 'Strict exposure-only on-demand. 12-month estimate = precompute once + 288 one-hour dispatches.'
	};

	const benGurionFullDay = results.find(
		(result) => result.timing.analysisId === 'Ben-Gurion/20250815_grid_2m_fullday'
	);
	const largestAnalysis = results.reduce((currentLargest, candidate) => {
		if (!currentLargest) return candidate;
		return candidate.numPoints > currentLargest.numPoints ? candidate : currentLargest;
	}, null as any);

	const headlineRows = [
		medianProfile,
		benGurionFullDay
			? {
					label: benGurionFullDay.timing.analysisId,
					points: benGurionFullDay.onDemand.pointCount ?? benGurionFullDay.numPoints,
					oneHourNowS: benGurionFullDay.onDemand.readyWallS,
					twelveMonthNowS: benGurionFullDay.onDemand.fullYearEstimateS,
					note: 'Main Ben-Gurion full-day baseline analysis.'
				}
			: null,
		largestAnalysis && largestAnalysis?.timing.analysisId !== benGurionFullDay?.timing.analysisId
			? {
					label: largestAnalysis.timing.analysisId,
					points: largestAnalysis.onDemand.pointCount ?? largestAnalysis.numPoints,
					oneHourNowS: largestAnalysis.onDemand.readyWallS,
					twelveMonthNowS: largestAnalysis.onDemand.fullYearEstimateS,
					note: 'Largest grid in the current batch.'
				}
			: null
	].filter(Boolean) as Array<{
		label: string;
		points: number | null;
		oneHourNowS: number | null;
		twelveMonthNowS: number | null;
		note: string;
	}>;

	let md = '# WebGPU Parity & Performance Report: Run-All vs On-Demand\n\n';
	md += `Generated on: ${new Date().toLocaleString()}\n\n`;
	md += `Legacy run-all-only report preserved at: \`${legacyReportPath}\`\n`;
	md += `New combined report written to: \`${reportPath}\`\n\n`;

	md += '## 1. Headline Speedup vs Grasshopper 1-Hour UTCI Baseline\n\n';
	md += '| Profile | Points | 1h GH Baseline | 1h On-Demand Now | 1h Speedup | 12m GH Estimate | 12m On-Demand Estimate | 12m Speedup | Notes |\n';
	md += '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n';
	for (const row of headlineRows) {
		md += `| ${row.label} | ${formatInt(row.points)} | ${formatDurationHuman(GRASSHOPPER_ONE_HOUR_BASELINE_S)} | ${formatDurationHuman(row.oneHourNowS)} | **${formatSpeedup(GRASSHOPPER_ONE_HOUR_BASELINE_S, row.oneHourNowS)}** | ${formatDurationHuman(grasshopperTwelveMonthBaselineS)} | ${formatDurationHuman(row.twelveMonthNowS)} | **${formatSpeedup(grasshopperTwelveMonthBaselineS, row.twelveMonthNowS)}** | ${row.note} |\n`;
	}
	md += '\n';
	md += `Assumptions: Grasshopper baseline is fixed at ~15 minutes (${GRASSHOPPER_ONE_HOUR_BASELINE_S}s) per UTCI hour. `;
	md += `The 12-month estimate uses this repo's representative full-year sweep of ${REPRESENTATIVE_FULL_YEAR_HOURS} hourly evaluations (12 months x 24 hours), not an 8,760-hour annual run. `;
	md += 'On-demand estimates come from strict exposure-only diagnostics: one exposure precompute plus repeated one-hour dispatches.\n\n';

	md += '## 2. Per-Analysis On-Demand Summary\n\n';
	md += '| Analysis | Points | State | Ready Wall (s) | Exposure Precompute (ms) | One-Hour Dispatch (ms) | 12m Estimate (s) | Persistent Exposure (MiB) | Selected-Hour HWM (KiB) | Zero All-Hours Alloc | Path |\n';
	md += '| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |\n';
	for (const result of results) {
		const onDemand = result.onDemand;
		const zeroAlloc =
			onDemand.zeroAllHoursAlloc == null ? '-' : onDemand.zeroAllHoursAlloc ? 'PASS' : 'FAIL';
		md += `| ${result.timing.analysisId} | ${formatInt(onDemand.pointCount ?? result.numPoints)} | ${onDemand.state} | ${formatFixed(onDemand.readyWallS, 2)} | ${formatFixed(onDemand.exposurePrecomputeMs, 1)} | ${formatFixed(onDemand.oneHourDispatchMs, 3)} | ${formatFixed(onDemand.fullYearEstimateS, 3)} | ${formatMiB(onDemand.persistentExposureBytes)} | ${formatKiB(onDemand.selectedHourOutputBytesHighWatermark)} | ${zeroAlloc} | ${onDemand.path ?? '-'} |\n`;
	}

	md += '\n## 3. Existing Run-All Route Summary\n\n';
	md += '| Analysis | Python Full-Day (s) | WebGPU Parity/Day (s) | Speedup vs Python | WebGPU Full-Year (s) | Speedup vs Python x12 | Solar | Sky | MRT | UTCI |\n';
	md += '| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |\n';
	for (const result of results) {
		const pythonTime = numberOrNull(result.timing.pythonRuntime);
		const parityDayTime = numberOrNull(result.timing.webgpu_1m?.compute_s);
		const fullYearTime = numberOrNull(result.timing.webgpu_12m?.compute_s);
		const fullYearBaseline = pythonTime != null ? pythonTime * 12 : null;
		md += `| ${result.timing.analysisId} | ${formatFixed(pythonTime, 1)} | ${formatFixed(parityDayTime, 2)} | **${formatSpeedup(pythonTime, parityDayTime)}** | ${formatFixed(fullYearTime, 2)} | **${formatSpeedup(fullYearBaseline, fullYearTime)}** | ${result.parityStatus.solar} | ${result.parityStatus.sky} | ${result.parityStatus.mrt} | ${result.parityStatus.utci} |\n`;
	}

	md += '\n## 4. Legacy "Store All" Memory Estimate\n\n';
	md += '| Analysis | Grid Points | Solar (MiB) | Sky (MiB) | Results (MiB) | Total VRAM (MiB) |\n';
	md += '| :--- | ---: | ---: | ---: | ---: | ---: |\n';
	for (const result of results) {
		md += `| ${result.timing.analysisId} | ${formatInt(result.numPoints)} | ${result.legacyStoreAll.solar.toFixed(1)} | ${result.legacyStoreAll.sky.toFixed(1)} | ${result.legacyStoreAll.utci.toFixed(1)} | **${result.legacyStoreAll.total.toFixed(1)}** |\n`;
	}

	md += '\n## 5. Detailed Full-Year Timing Breakdown\n\n';
	md += '| Analysis | Weather | Init | BVH | GPU Compute | Readback/Wait | Total (s) |\n';
	md += '| :--- | :--- | :--- | :--- | :--- | :--- | ---: |\n';
	for (const result of results) {
		const formatPhase = (phase: any) => `${phase.s.toFixed(3)}s (${phase.pct})`;
		md += `| ${result.timing.analysisId} | ${formatPhase(result.durFullYear.weather)} | ${formatPhase(result.durFullYear.init)} | ${formatPhase(result.durFullYear.bvh)} | **${formatPhase(result.durFullYear.compute)}** | **${formatPhase(result.durFullYear.readback)}** | **${result.durFullYear.total.s.toFixed(2)}** |\n`;
	}

	md += '\n## 6. Detailed Parity/Day Timing Breakdown\n\n';
	md += '| Analysis | Weather | Init | BVH | GPU Compute | Readback/Wait | Total (s) |\n';
	md += '| :--- | :--- | :--- | :--- | :--- | :--- | ---: |\n';
	for (const result of results) {
		const formatPhase = (phase: any) => `${phase.s.toFixed(3)}s (${phase.pct})`;
		md += `| ${result.timing.analysisId} | ${formatPhase(result.durParityDay.weather)} | ${formatPhase(result.durParityDay.init)} | ${formatPhase(result.durParityDay.bvh)} | **${formatPhase(result.durParityDay.compute)}** | **${formatPhase(result.durParityDay.readback)}** | **${result.durParityDay.total.s.toFixed(2)}** |\n`;
	}

	writeFileSync(reportPath, md);
	console.log(`Report generated at ${reportPath}`);
	console.log(`Legacy report preserved at ${legacyReportPath}`);
}

main().catch(console.error);
