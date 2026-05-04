import { resolve, join } from 'node:path';
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { execSync } from 'node:child_process';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const resultsDir = resolve(REPO_ROOT, 'data/batch-parity-results');
const reportPath = resolve(REPO_ROOT, 'data/batch-parity-results/parity_performance_report.md');

async function main() {
	if (!existsSync(resultsDir)) {
		console.error('Results directory not found');
		return;
	}

	const files = readdirSync(resultsDir).filter(f => f.endsWith('_timing.json'));
	console.log(`Found ${files.length} timing files`);

	let md = '# WebGPU vs Python Parity & Performance Report\n\n';
	md += `Generated on: ${new Date().toLocaleString()}\n\n`;
	md += '| Analysis | Python (s) | WebGPU 1m (s) | Speedup 1m | WebGPU 12m (s) | Speedup 12m | Solar | Sky | MRT | UTCI | Collect (s) |\n';
	md += '| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: | ---: |\n';

	for (const file of files) {
		const timing = JSON.parse(readFileSync(join(resultsDir, file), 'utf8'));
		const analysisId = timing.analysisId;
		const pythonTime = timing.pythonRuntime || 0;
		
		const t1m = timing.webgpu_1m?.compute_s || 0;
		const t12m = timing.webgpu_12m?.compute_s || 0;
		
		const speedup1m = (pythonTime > 0 && t1m > 0) ? (pythonTime / t1m).toFixed(1) + 'x' : '-';
		const speedup12m = (pythonTime > 0 && t12m > 0) ? (pythonTime * 12 / t12m).toFixed(1) + 'x' : '-'; // Python is only 1 month, so multiply by 12 for fair comparison
		
		const collectTime = (timing.webgpu_1m?.collect_s || 0).toFixed(1);

		console.log(`Analyzing parity for ${analysisId}...`);
		let solar = '-', sky = '-', mrt = '-', utci = '-';

		try {
			// Run offline parity comparison (always uses 1-month data)
			const compareCmd = `npx tsx scripts/compare-parity.ts --base-path data/analyses/${analysisId} --mode stats`;
			let output = '';
			try {
				output = execSync(compareCmd, { cwd: resolve(REPO_ROOT, 'viewer'), encoding: 'utf8' });
			} catch (err: any) {
				output = err.stdout || '';
			}
			
			const parseStatus = (layer: string) => {
				if (output.includes(`${layer}: PASS`)) return '✅';
				if (output.includes(`${layer}: FAIL (no ref or error)`)) return '-';
				if (output.includes(`${layer}: FAIL`)) return '❌';
				return '-';
			};

			solar = parseStatus('solar');
			sky = parseStatus('sky');
			mrt = parseStatus('mrt');
			utci = parseStatus('utci');
		} catch (err: any) {
			console.warn(`Comparison crashed for ${analysisId}: ${err.message}`);
		}

		md += `| ${analysisId} | ${pythonTime.toFixed(1)} | ${t1m.toFixed(2)} | **${speedup1m}** | ${t12m.toFixed(2)} | **${speedup12m}** | ${solar} | ${sky} | ${mrt} | ${utci} | ${collectTime} |\n`;
	}

	writeFileSync(reportPath, md);
	console.log(`Report generated at ${reportPath}`);
}

main().catch(console.error);
