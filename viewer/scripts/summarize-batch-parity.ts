import { resolve, join } from 'node:path';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';

const execAsync = promisify(exec);
const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const resultsDir = resolve(REPO_ROOT, 'data/batch-parity-results');
const reportPath = resolve(REPO_ROOT, 'data/batch-parity-results/parity_performance_report.md');

// Limit concurrency to avoid overloading the system
const CONCURRENCY = Math.max(1, Math.min(os.cpus().length - 1, 8));

async function main() {
	if (!existsSync(resultsDir)) {
		console.error('Results directory not found');
		return;
	}

	const files = readdirSync(resultsDir).filter(f => f.endsWith('_timing.json'));
	console.log(`Found ${files.length} timing files. Processing with concurrency ${CONCURRENCY}...`);

	const results: any[] = [];
	const queue = [...files];
	const activePromises: Promise<void>[] = [];

	const processFile = async (file: string) => {
		const timing = JSON.parse(readFileSync(join(resultsDir, file), 'utf8'));
		const analysisId = timing.analysisId;
		console.log(`[${results.length + 1}/${files.length}] Analyzing ${analysisId}...`);

		let parityStatus = { solar: '-', sky: '-', mrt: '-', utci: '-' };
		try {
			const compareCmd = `npx tsx scripts/compare-parity.ts --base-path data/analyses/${analysisId} --mode stats`;
			let output = '';
			try {
				const { stdout } = await execAsync(compareCmd, { cwd: resolve(REPO_ROOT, 'viewer') });
				output = stdout;
			} catch (err: any) {
				output = err.stdout || '';
			}
			
			const parseStatus = (layer: string) => {
				if (output.includes(`${layer}: PASS`)) return '✅';
				if (output.includes(`${layer}: FAIL (no ref or error)`)) return '-';
				if (output.includes(`${layer}: FAIL`)) return '❌';
				return '-';
			};
			parityStatus = {
				solar: parseStatus('solar'),
				sky: parseStatus('sky'),
				mrt: parseStatus('mrt'),
				utci: parseStatus('utci')
			};
		} catch (err) {
			console.warn(`Comparison failed for ${analysisId}`);
		}

		const calculateDurations = (log: any[]) => {
			const getStep = (phase: string) => log?.find((l: any) => l.phase === phase)?.timestamp;
			
			const tStart = getStep('preflight');
			const tEpw = getStep('epw');
			const tInit = getStep('pipelineInit');
			const tWorker = getStep('worker');
			const tRun = getStep('runAll');
			const tReadback = getStep('readback');
			const tDone = getStep('done');

			const dur = {
				weather: (tInit && tEpw) ? (tInit - tEpw) : 0,
				init: (tWorker && tInit) ? (tWorker - tInit) : 0,
				bvh: (tRun && tWorker) ? (tRun - tWorker) : 0,
				compute: (tReadback && tRun) ? (tReadback - tRun) : 0,
				readback: (tDone && tReadback) ? (tDone - tReadback) : 0,
				total: (tDone && tStart) ? (tDone - tStart) : 0
			};
			
			const total = Math.max(0.001, dur.total);
			const pct = (v: number) => ((v / total) * 100).toFixed(1) + '%';

			return {
				weather: { s: dur.weather / 1000, pct: pct(dur.weather) },
				init: { s: dur.init / 1000, pct: pct(dur.init) },
				bvh: { s: dur.bvh / 1000, pct: pct(dur.bvh) },
				compute: { s: dur.compute / 1000, pct: pct(dur.compute) },
				readback: { s: dur.readback / 1000, pct: pct(dur.readback) },
				total: { s: dur.total / 1000 }
			};
		};

		const dur1m = calculateDurations(timing.webgpu_1m?.log);
		const dur12m = calculateDurations(timing.webgpu_12m?.log);

		// Memory Estimation (current "Store All" arch)
		const numPoints = timing.preflight?.numPoints ?? timing.preflight?.estimatedGridPoints ?? 0;
		const mem = {
			solar: (numPoints * 24 * 12 / 8) / (1024 * 1024), // bit-packed
			sky: (numPoints * 145 * 4) / (1024 * 1024), // Float32
			utci: (numPoints * 24 * 12 * 4) / (1024 * 1024), // Float32 storage
			total: 0
		};
		mem.total = mem.solar + mem.sky + mem.utci;

		results.push({ timing, parityStatus, dur1m, dur12m, mem });
	};

	// Simple worker pool
	const startWorker = async () => {
		while (queue.length > 0) {
			const file = queue.shift();
			if (file) await processFile(file);
		}
	};

	for (let i = 0; i < CONCURRENCY; i++) {
		activePromises.push(startWorker());
	}
	await Promise.all(activePromises);

	// Generate Report
	let md = '# WebGPU vs Python Parity & Performance Report\n\n';
	md += `Generated on: ${new Date().toLocaleString()}\n\n`;
	
	md += '## 1. Summary Comparison\n\n';
	md += '| Analysis | Python 1m (s) | WebGPU 1m (s) | Speedup 1m | WebGPU 12m (s) | Speedup 12m | Solar | Sky | MRT | UTCI |\n';
	md += '| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: |\n';

	for (const res of results) {
		const { timing, parityStatus } = res;
		const pythonTime = timing.pythonRuntime || 0;
		const t1m = timing.webgpu_1m?.compute_s || 0;
		const t12m = timing.webgpu_12m?.compute_s || 0;
		const speedup1m = (pythonTime > 0 && t1m > 0) ? (pythonTime / t1m).toFixed(1) + 'x' : '-';
		const speedup12m = (pythonTime > 0 && t12m > 0) ? (pythonTime * 12 / t12m).toFixed(1) + 'x' : '-';
		
		md += `| ${timing.analysisId} | ${pythonTime.toFixed(1)} | ${t1m.toFixed(2)} | **${speedup1m}** | ${t12m.toFixed(2)} | **${speedup12m}** | ${parityStatus.solar} | ${parityStatus.sky} | ${parityStatus.mrt} | ${parityStatus.utci} |\n`;
	}

	md += '\n## 2. Memory Usage (Current "Store All" Arch)\n\n';
	md += '| Analysis | Grid Points | Solar (MB) | Sky (MB) | Results (MB) | **Total VRAM (MB)** |\n';
	md += '| :--- | ---: | ---: | ---: | ---: | ---: |\n';

	for (const res of results) {
		const { timing, mem } = res;
		const numPoints = timing.preflight?.numPoints ?? timing.preflight?.estimatedGridPoints ?? 0;
		md += `| ${timing.analysisId} | ${numPoints.toLocaleString()} | ${mem.solar.toFixed(1)} | ${mem.sky.toFixed(1)} | ${mem.utci.toFixed(1)} | **${mem.total.toFixed(1)}** |\n`;
	}

	md += '\n## 3. Detailed 12-Month Timing Breakdown\n\n';
	md += '| Analysis | Weather | Init | BVH | **GPU Compute** | **Readback/Wait** | **Total (s)** |\n';
	md += '| :--- | :--- | :--- | :--- | :--- | :--- | ---: |\n';

	for (const res of results) {
		const { dur12m, timing } = res;
		const f = (d: any) => `${d.s.toFixed(3)}s (${d.pct})`;
		md += `| ${timing.analysisId} | ${f(dur12m.weather)} | ${f(dur12m.init)} | ${f(dur12m.bvh)} | **${f(dur12m.compute)}** | **${f(dur12m.readback)}** | **${dur12m.total.s.toFixed(2)}** |\n`;
	}

	md += '\n## 4. Detailed 1-Month Timing Breakdown\n\n';
	md += '| Analysis | Weather | Init | BVH | **GPU Compute** | **Readback/Wait** | **Total (s)** |\n';
	md += '| :--- | :--- | :--- | :--- | :--- | :--- | ---: |\n';

	for (const res of results) {
		const { dur1m, timing } = res;
		const f = (d: any) => `${d.s.toFixed(3)}s (${d.pct})`;
		md += `| ${timing.analysisId} | ${f(dur1m.weather)} | ${f(dur1m.init)} | ${f(dur1m.bvh)} | **${f(dur1m.compute)}** | **${f(dur1m.readback)}** | **${dur1m.total.s.toFixed(2)}** |\n`;
	}

	writeFileSync(reportPath, md);
	console.log(`Report generated at ${reportPath}`);
}

main().catch(console.error);
