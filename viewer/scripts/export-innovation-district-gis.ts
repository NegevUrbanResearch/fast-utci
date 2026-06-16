import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { chromium, type Browser, type Page } from '@playwright/test';

import {
	buildActiveCellArrays,
	buildRawExportMetadata
} from '../src/lib/gis/innovationDistrictExport';
import { resolveRepoRelativePath } from './repo-paths';

const ANALYSIS_ID = 'Innovation-District/innovation_district_webgpu';
const DEFAULT_PORT = 4173;
const DEFAULT_DATE = '2025-08-15';
const DEFAULT_MONTH_INDEX = 7;
const GRID_SIZE_METERS = 2;
const DEFAULT_HOURS = Object.freeze(Array.from({ length: 24 }, (_, index) => index));
const COLLECTOR_QUERY_FLAG = 'fastUtciCollectorExport';
const RAW_SCHEMA_VERSION = 'innovation-district-raw-export/v1';
const SOURCE_GEOREF_PATH = 'data/3d_models/Innovation-District/innovation_district.georef.json';

type MetricType = 'utci' | 'shading_index';

type CollectorExportResult = {
	metadata: {
		analysisId: string;
		metricType: MetricType;
		monthIndex: number;
		hourIndex: number;
		timeIndex: number;
		activeMask: {
			source: 'base+road';
			activeCanonicalIndices: Uint32Array | number[];
			canonicalPointCount: number;
			activePointCount: number;
			inactivePointCount: number;
			activePointRatio: number;
			checksum: string;
			signature: string;
		};
	};
	canonicalIndices: Uint32Array | number[];
	positions: Float32Array | number[];
	values: Float32Array | number[];
};

type CollectorDiagnosticsSnapshot = {
	utciRenderResolved?: string;
	rendererBackend?: string;
	baseRenderTransport?: string;
	baseLiveReady?: boolean;
	selectedHourRuntimeContract?: {
		renderTransport?: string;
		utciSurfaceSource?: string;
	};
} & Record<string, unknown>;

type Args = {
	outDir: string;
	date: string;
	monthIndex: number;
	hours: number[];
	port: number;
	baseUrl?: string;
	headless: boolean;
	dryRunArgs: boolean;
	devServerSmoke: boolean;
};

type TimingBreakdown = {
	routeLoad: number;
	liveSessionReady: number;
	utciCollection: number;
	shadingCollection: number;
	binarySerialization: number;
	total: number;
};

type ParseExportArgsOptions = {
	argv?: string[];
	cwd?: string;
	env?: NodeJS.ProcessEnv;
};

type RawExportFileNames = {
	metadata: string;
	canonicalIndices: string;
	positions: string;
	utci: string;
	shadingIndex: string;
};

function resolveRepoRelativeOutDir(cwd: string, outDir: string): string {
	return resolveRepoRelativePath(cwd, outDir);
}

export function parseExportArgs(options: ParseExportArgsOptions = {}): Args {
	const env = options.env ?? process.env;
	const cwd = options.cwd ?? process.cwd();
	const argv = normalizeNpmForwardedArgs(options.argv ?? process.argv.slice(2), env);
	let outDir = '';
	let date = DEFAULT_DATE;
	let monthIndex = DEFAULT_MONTH_INDEX;
	let hours = [...DEFAULT_HOURS];
	let port = DEFAULT_PORT;
	let baseUrl: string | undefined;
	let headless = false;
	let dryRunArgs = false;
	let devServerSmoke = false;

	for (let i = 0; i < argv.length; i += 1) {
		const arg = argv[i];
		const next = argv[i + 1];
		if (arg === '--out-dir' && next) {
			outDir = next;
			i += 1;
		} else if (arg === '--date' && next) {
			date = next;
			i += 1;
		} else if (arg === '--month-index' && next) {
			monthIndex = Number(next);
			i += 1;
		} else if (arg === '--hour' && next) {
			hours = [Number(next)];
			i += 1;
		} else if (arg === '--hours' && next) {
			hours = parseHours(next);
			i += 1;
		} else if (arg === '--port' && next) {
			port = Number(next);
			i += 1;
		} else if (arg === '--base-url' && next) {
			baseUrl = next;
			i += 1;
		} else if (arg === '--headless') {
			headless = true;
		} else if (arg === '--dry-run-args') {
			dryRunArgs = true;
		} else if (arg === '--dev-server-smoke') {
			devServerSmoke = true;
		} else if (!arg.startsWith('-') && !outDir) {
			outDir = arg;
		} else {
			throw new Error(`Unknown or incomplete argument: ${arg}`);
		}
	}

	if (!outDir) {
		throw new Error('Missing --out-dir <directory>.');
	}
	if (!Number.isInteger(monthIndex) || monthIndex < 0 || monthIndex > 11) {
		throw new Error(`Invalid --month-index ${monthIndex}; expected 0-11.`);
	}
	if (!Number.isInteger(port) || port <= 0) {
		throw new Error(`Invalid --port ${port}.`);
	}
	if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
		throw new Error(`Invalid --date ${date}; expected YYYY-MM-DD.`);
	}
	for (const hour of hours) {
		if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
			throw new Error(`Invalid hour ${hour}; expected 0-23.`);
		}
	}

	return {
		outDir: resolveRepoRelativeOutDir(cwd, outDir),
		date,
		monthIndex,
		hours,
		port,
		baseUrl,
		headless,
		dryRunArgs,
		devServerSmoke
	};
}

export function buildRawExportFileNames(params: {
	date: string;
	gridSize: number;
}): RawExportFileNames {
	const prefix = `${params.date}_${params.gridSize}m_active-cells`;
	return {
		metadata: `${prefix}.metadata.json`,
		canonicalIndices: `${prefix}.canonical.u32.bin`,
		positions: `${prefix}.positions.f32.bin`,
		utci: `${prefix}.utci.f32.bin`,
		shadingIndex: `${prefix}.shading.f32.bin`
	};
}

function normalizeNpmForwardedArgs(argv: string[], env: NodeJS.ProcessEnv): string[] {
	if (!env.npm_lifecycle_event || !hasNpmConfigExportArgs(env)) return argv;

	const normalized: string[] = [];
	const strippedValues = argv.filter((arg) => !arg.startsWith('-'));
	pushConfigFlag(normalized, '--dry-run-args', env.npm_config_dry_run_args);
	pushConfigFlag(normalized, '--dev-server-smoke', env.npm_config_dev_server_smoke);
	pushConfigValue(normalized, '--out-dir', env.npm_config_out_dir, strippedValues);
	pushConfigValue(normalized, '--date', env.npm_config_date, strippedValues);
	pushConfigValue(normalized, '--month-index', env.npm_config_month_index, strippedValues);
	pushConfigValue(normalized, '--hour', env.npm_config_hour, strippedValues);
	pushConfigValue(normalized, '--hours', env.npm_config_hours, strippedValues);
	pushConfigValue(normalized, '--port', env.npm_config_port, strippedValues);
	pushConfigValue(normalized, '--base-url', env.npm_config_base_url, strippedValues);
	pushConfigFlag(normalized, '--headless', env.npm_config_headless);

	return normalized.length > 0 ? normalized : argv;
}

function hasNpmConfigExportArgs(env: NodeJS.ProcessEnv): boolean {
	return [
		'npm_config_dry_run_args',
		'npm_config_dev_server_smoke',
		'npm_config_out_dir',
		'npm_config_date',
		'npm_config_month_index',
		'npm_config_hour',
		'npm_config_hours',
		'npm_config_port',
		'npm_config_base_url',
		'npm_config_headless'
	].some((key) => env[key] != null);
}

function pushConfigValue(
	args: string[],
	flag: string,
	value: string | undefined,
	strippedValues: string[]
): void {
	if (value == null || value === '') return;
	args.push(flag, value === 'true' ? (strippedValues.shift() ?? value) : value);
}

function pushConfigFlag(args: string[], flag: string, value: string | undefined): void {
	if (value == null || value === '' || value === 'false') return;
	args.push(flag);
}

function parseHours(value: string): number[] {
	const range = value.match(/^(\d+)-(\d+)$/);
	if (range) {
		const start = Number(range[1]);
		const end = Number(range[2]);
		if (end < start) throw new Error(`Invalid --hours range ${value}.`);
		return Array.from({ length: end - start + 1 }, (_, index) => start + index);
	}
	return value.split(',').map((part) => Number(part.trim()));
}

function nowMs(): number {
	return performance.now();
}

async function isServerReachable(baseUrl: string): Promise<boolean> {
	try {
		const response = await fetch(baseUrl, { method: 'GET' });
		return response.ok;
	} catch {
		return false;
	}
}

async function startViteIfNeeded(args: Args, baseUrl: string): Promise<ChildProcess | null> {
	if (await isServerReachable(baseUrl)) {
		return null;
	}

	const viteBin = resolve(process.cwd(), 'node_modules', 'vite', 'bin', 'vite.js');
	const child = spawn(process.execPath, [viteBin, 'dev', '--host', '127.0.0.1', '--port', String(args.port)], {
		cwd: process.cwd(),
		stdio: ['ignore', 'pipe', 'pipe'],
		shell: false,
		detached: process.platform !== 'win32',
		env: { ...process.env }
	});
	child.stdout?.on('data', (chunk) => process.stdout.write(chunk));
	child.stderr?.on('data', (chunk) => process.stderr.write(chunk));

	const startedAt = Date.now();
	while (Date.now() - startedAt < 120_000) {
		if (await isServerReachable(baseUrl)) {
			return child;
		}
		await new Promise((resolvePromise) => setTimeout(resolvePromise, 500));
	}
	await stopProcessTree(child);
	throw new Error(`Timed out waiting for Vite dev server at ${baseUrl}.`);
}

async function stopProcessTree(child: ChildProcess): Promise<void> {
	if (child.exitCode != null || child.signalCode != null || child.pid == null) return;

	if (process.platform === 'win32') {
		await new Promise<void>((resolvePromise) => {
			execFile(
				'taskkill',
				['/pid', String(child.pid), '/t', '/f'],
				{ windowsHide: true },
				() => resolvePromise()
			);
		});
		await waitForExit(child, 5_000);
		return;
	}

	try {
		process.kill(-child.pid, 'SIGTERM');
	} catch {
		child.kill('SIGTERM');
	}
	const exited = await waitForExit(child, 2_000);
	if (exited) return;
	try {
		process.kill(-child.pid, 'SIGKILL');
	} catch {
		child.kill('SIGKILL');
	}
	await waitForExit(child, 2_000);
}

function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
	if (child.exitCode != null || child.signalCode != null) return Promise.resolve(true);
	return new Promise((resolvePromise) => {
		const timeout = setTimeout(() => {
			child.off('exit', onExit);
			resolvePromise(false);
		}, timeoutMs);
		const onExit = () => {
			clearTimeout(timeout);
			resolvePromise(true);
		};
		child.once('exit', onExit);
	});
}

function routeUrl(baseUrl: string): string {
	const url = new URL('/', baseUrl);
	url.searchParams.set('analysis', ANALYSIS_ID);
	url.searchParams.set('gridResolution', String(GRID_SIZE_METERS));
	url.searchParams.set('utciRender', 'auto');
	url.searchParams.set('utciRenderDiagnostics', '1');
	url.searchParams.set(COLLECTOR_QUERY_FLAG, '1');
	return url.toString();
}

function isHeadlessWebGpuPublicationUnsupported(params: {
	headless: boolean;
	diagnostics: CollectorDiagnosticsSnapshot | null;
}): boolean {
	if (!params.headless || params.diagnostics == null) return false;
	const runtimeContract = params.diagnostics.selectedHourRuntimeContract;
	return (
		params.diagnostics.utciRenderResolved === 'dataTexture' &&
		params.diagnostics.rendererBackend === 'unknown' &&
		params.diagnostics.baseRenderTransport === 'idle' &&
		params.diagnostics.baseLiveReady === false &&
		runtimeContract?.renderTransport === 'none' &&
		runtimeContract?.utciSurfaceSource === 'none'
	);
}

export function buildPublicationTimeoutErrorMessage(params: {
	metricType: MetricType;
	headless: boolean;
	error: unknown;
	diagnostics: CollectorDiagnosticsSnapshot | null;
}): string {
	const lines = [
		`Timed out waiting for ${params.metricType} publication.`,
		params.error instanceof Error ? params.error.message : String(params.error)
	];

	if (isHeadlessWebGpuPublicationUnsupported(params)) {
		lines.push(
			'Headless Chromium did not expose the required live WebGPU publication contract in this environment.',
			'The collector only writes truthful raw exports from the gpuNative / compute-buffer-selected-hour path.',
			'Rerun without --headless so the viewer can use the real live WebGPU route state.'
		);
	}

	lines.push(
		'Last window.__utciRenderDiagnostics__:',
		JSON.stringify(params.diagnostics, null, 2)
	);
	return lines.join('\n');
}

async function waitForPublication(
	page: Page,
	params: {
		metricType: MetricType;
		monthIndex: number;
		hourIndex: number;
		previousRequestId?: number;
		headless: boolean;
	}
): Promise<Record<string, unknown>> {
	const expectedSelectionKey =
		params.metricType === 'shading_index'
			? `${ANALYSIS_ID}|shading_index|${params.monthIndex}`
			: `${ANALYSIS_ID}|utci|${params.monthIndex}|${params.hourIndex}`;

	const handle = await page
		.waitForFunction(
			(args) => {
				const value = (window as any).__utciRenderDiagnostics__;
				const timeline = value?.timings?.renderPublication?.renderPublicationTimeline;
				if (!value) return null;
				const surfaceRequestId = value.baseSurfaceRequestId;
				return value.rendererBackend === 'webgpu' &&
					value.utciRenderRequested === 'auto' &&
					value.utciRenderResolved === 'gpuNative' &&
					value.baseLiveReady === true &&
					value.baseRenderTransport === 'compute-buffer-selected-hour' &&
					value.utciSurfaceSource === 'compute-buffer-selected-hour' &&
					value.baseSameDeviceForComputeAndRender === true &&
					value.baseSelectionKey === args.expectedSelectionKey &&
					value.baseSceneSelectionKey === args.expectedSelectionKey &&
					value.baseSelectedMonthIndex === args.monthIndex &&
					value.baseSelectedHourIndex === args.expectedHourIndex &&
					value.baseSelectedTimeIndex === args.monthIndex * 24 + args.expectedHourIndex &&
					typeof surfaceRequestId === 'number' &&
					(typeof args.previousRequestId !== 'number' ||
						surfaceRequestId > args.previousRequestId) &&
					value.gpuResidentCopyRequestId === surfaceRequestId &&
					value.renderAllocationPreflight?.renderTopology === 'active-cells' &&
					value.renderAllocationPreflight?.activePointCount > 0 &&
					value.renderAllocationPreflight?.canonicalCellCount >
						value.renderAllocationPreflight?.activePointCount &&
					timeline?.sessionMetricType === args.metricType
					? value
					: null;
			},
			{
				metricType: params.metricType,
				monthIndex: params.monthIndex,
				expectedHourIndex: params.metricType === 'shading_index' ? 0 : params.hourIndex,
				expectedSelectionKey,
				previousRequestId: params.previousRequestId
			},
			{ timeout: 180_000 }
		)
		.catch(async (error) => {
			const diagnostics = (await page.evaluate(
				() => (window as any).__utciRenderDiagnostics__ ?? null
			)) as CollectorDiagnosticsSnapshot | null;
			throw new Error(
				buildPublicationTimeoutErrorMessage({
					metricType: params.metricType,
					headless: params.headless,
					error,
					diagnostics
				})
			);
		});
	return handle.jsonValue() as Promise<Record<string, unknown>>;
}

async function setMetricSelection(
	page: Page,
	params: { metricType: MetricType; monthIndex: number; hourIndex: number }
): Promise<void> {
	await page.evaluate(async (selection) => {
		const store = (await new Function(
			'return import("/src/lib/stores/viewerStore.ts")'
		)()) as typeof import('../src/lib/stores/viewerStore');
		store.setCurrentMonth(selection.monthIndex);
		store.setCurrentHour(selection.hourIndex);
		store.setMetricType(selection.metricType);
		await new Promise<void>((resolvePromise) => requestAnimationFrame(() => resolvePromise()));
	}, params);
}

async function collectCurrent(page: Page): Promise<CollectorExportResult> {
	return page.evaluate(async () => {
		const exportFn = (window as any).__fastUtciCollectorExport;
		if (typeof exportFn !== 'function') {
			throw new Error('window.__fastUtciCollectorExport is not available.');
		}
		return exportFn();
	});
}

function toUint32Array(value: Uint32Array | number[], name: string): Uint32Array {
	if (value instanceof Uint32Array) return new Uint32Array(value);
	if (Array.isArray(value)) return new Uint32Array(value);
	throw new Error(`${name} did not serialize as a Uint32Array.`);
}

function toFloat32Array(value: Float32Array | number[], name: string): Float32Array {
	if (value instanceof Float32Array) return new Float32Array(value);
	if (Array.isArray(value)) return new Float32Array(value);
	throw new Error(`${name} did not serialize as a Float32Array.`);
}

function arraysEqual(left: Uint32Array, right: Uint32Array): boolean {
	if (left.length !== right.length) return false;
	for (let i = 0; i < left.length; i += 1) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}

function sha256(array: Uint32Array | Float32Array): string {
	return createHash('sha256')
		.update(Buffer.from(array.buffer, array.byteOffset, array.byteLength))
		.digest('hex');
}

async function writeBinary(path: string, array: Uint32Array | Float32Array): Promise<void> {
	await writeFile(path, Buffer.from(array.buffer, array.byteOffset, array.byteLength));
}

async function loadDeclaredCrs(): Promise<string> {
	const absolute = resolveRepoRelativePath(process.cwd(), SOURCE_GEOREF_PATH);
	if (!existsSync(absolute)) return 'unknown';
	const parsed = JSON.parse(await readFile(absolute, 'utf8')) as { declared_crs?: string };
	return parsed.declared_crs ?? 'unknown';
}

async function main(): Promise<void> {
	const totalStartedAt = nowMs();
	const args = parseExportArgs();
	if (args.dryRunArgs) {
		console.log(
			JSON.stringify(
				{
					outDir: args.outDir,
					date: args.date,
					monthIndex: args.monthIndex,
					hours: args.hours,
					port: args.port,
					baseUrl: args.baseUrl,
					headless: args.headless,
					devServerSmoke: args.devServerSmoke
				},
				null,
				2
			)
		);
		return;
	}
	const baseUrl = args.baseUrl ?? `http://localhost:${args.port}`;
	const timings: Partial<TimingBreakdown> = {};
	let server: ChildProcess | null = null;
	let browser: Browser | null = null;

	try {
		server = await startViteIfNeeded(args, baseUrl);
		if (args.devServerSmoke) {
			console.log(
				JSON.stringify(
					{
						devServerSmoke: true,
						startedServer: server != null,
						baseUrl,
						port: args.port
					},
					null,
					2
				)
			);
			return;
		}
		browser = await chromium.launch({
			headless: args.headless,
			args: ['--enable-unsafe-webgpu']
		});
		const page = await browser.newPage();
		page.setDefaultTimeout(180_000);

		const routeLoadStartedAt = nowMs();
		await page.goto(routeUrl(baseUrl), { waitUntil: 'domcontentloaded' });
		timings.routeLoad = nowMs() - routeLoadStartedAt;

		const readinessStartedAt = nowMs();
		await setMetricSelection(page, {
			metricType: 'utci',
			monthIndex: args.monthIndex,
			hourIndex: args.hours[0] ?? 0
		});
		const initialDiagnostics = await waitForPublication(page, {
			metricType: 'utci',
			monthIndex: args.monthIndex,
			hourIndex: args.hours[0] ?? 0,
			headless: args.headless
		});
		timings.liveSessionReady = nowMs() - readinessStartedAt;

		const utciStartedAt = nowMs();
		const utciByHour: Float32Array[] = [];
		let canonicalIndices: Uint32Array | null = null;
		let positions: Float32Array | null = null;
		let activeMask:
			| CollectorExportResult['metadata']['activeMask']
			| null = null;
		let previousRequestId = Number(initialDiagnostics.baseSurfaceRequestId ?? 0);
		const diagnosticsByHour: Record<string, unknown>[] = [];

		for (const hourIndex of args.hours) {
			await setMetricSelection(page, {
				metricType: 'utci',
				monthIndex: args.monthIndex,
				hourIndex
			});
			const diagnostics = await waitForPublication(page, {
				metricType: 'utci',
				monthIndex: args.monthIndex,
				hourIndex,
				previousRequestId: hourIndex === args.hours[0] ? undefined : previousRequestId,
				headless: args.headless
			});
			previousRequestId = Number(diagnostics.baseSurfaceRequestId ?? previousRequestId);
			diagnosticsByHour.push(diagnostics);

			const result = await collectCurrent(page);
			const nextCanonical = toUint32Array(result.canonicalIndices, 'canonicalIndices');
			const nextPositions = toFloat32Array(result.positions, 'positions');
			const values = toFloat32Array(result.values, 'utci values');
			if (result.metadata.metricType !== 'utci') {
				throw new Error(`Expected UTCI export, got ${result.metadata.metricType}.`);
			}
			if (canonicalIndices == null) {
				canonicalIndices = nextCanonical;
				positions = nextPositions;
				activeMask = result.metadata.activeMask;
			} else if (!arraysEqual(canonicalIndices, nextCanonical)) {
				throw new Error(`UTCI hour ${hourIndex} active canonical row order changed.`);
			}
			utciByHour.push(values);
		}
		timings.utciCollection = nowMs() - utciStartedAt;

		const shadingStartedAt = nowMs();
		await setMetricSelection(page, {
			metricType: 'shading_index',
			monthIndex: args.monthIndex,
			hourIndex: 0
		});
		const shadingDiagnostics = await waitForPublication(page, {
			metricType: 'shading_index',
			monthIndex: args.monthIndex,
			hourIndex: 0,
			previousRequestId,
			headless: args.headless
		});
		const shadingResult = await collectCurrent(page);
		const shadingCanonical = toUint32Array(shadingResult.canonicalIndices, 'shading canonicalIndices');
		if (canonicalIndices == null || positions == null || activeMask == null) {
			throw new Error('UTCI collection did not produce active cell arrays.');
		}
		if (!arraysEqual(canonicalIndices, shadingCanonical)) {
			throw new Error('Shading Index active canonical row order does not match UTCI.');
		}
		const shadingIndex = toFloat32Array(shadingResult.values, 'shading values');
		timings.shadingCollection = nowMs() - shadingStartedAt;

		const serializationStartedAt = nowMs();
		const arrays = buildActiveCellArrays({
			activeCanonicalIndices: canonicalIndices,
			positions,
			utciByHour,
			shadingIndex,
			hours: args.hours,
			activeMaskSource: activeMask.source
		});

		await mkdir(args.outDir, { recursive: true });
		const fileNames = buildRawExportFileNames({
			date: args.date,
			gridSize: GRID_SIZE_METERS
		});
		await writeBinary(resolve(args.outDir, fileNames.canonicalIndices), arrays.canonicalIndices);
		await writeBinary(resolve(args.outDir, fileNames.positions), arrays.positions);
		await writeBinary(resolve(args.outDir, fileNames.utci), arrays.utci);
		await writeBinary(resolve(args.outDir, fileNames.shadingIndex), arrays.shadingIndex);

		const fileChecksums = {
			canonicalIndices: sha256(arrays.canonicalIndices),
			positions: sha256(arrays.positions),
			utci: sha256(arrays.utci),
			shadingIndex: sha256(arrays.shadingIndex)
		};
		timings.binarySerialization = nowMs() - serializationStartedAt;
		timings.total = nowMs() - totalStartedAt;

		const declaredCrs = await loadDeclaredCrs();
		const metadata = buildRawExportMetadata({
			schemaVersion: RAW_SCHEMA_VERSION,
			analysisId: `${ANALYSIS_ID}:${args.date}`,
			sourceAnalysisId: ANALYSIS_ID,
			sourceModelPath: 'data/3d_models/Innovation-District/innovation_district.glb',
			sourceGeorefPath: SOURCE_GEOREF_PATH,
			declaredCrs,
			gridSize: GRID_SIZE_METERS,
			coordinateSystem: 'projected-analysis',
			hours: args.hours,
			canonicalIndices: arrays.canonicalIndices,
			positions: arrays.positions,
			utci: arrays.utci,
			shadingIndex: arrays.shadingIndex,
			activeMask: {
				source: activeMask.source,
				canonicalPointCount: activeMask.canonicalPointCount,
				checksum: activeMask.checksum,
				signature: activeMask.signature
			},
			files: {
				canonicalIndices: {
					fileName: fileNames.canonicalIndices,
					checksum: fileChecksums.canonicalIndices
				},
				positions: {
					fileName: fileNames.positions,
					checksum: fileChecksums.positions
				},
				utci: {
					fileName: fileNames.utci,
					checksum: fileChecksums.utci
				},
				shadingIndex: {
					fileName: fileNames.shadingIndex,
					checksum: fileChecksums.shadingIndex
				}
			},
			timingsMs: timings as TimingBreakdown
		});

		const metadataPath = resolve(args.outDir, fileNames.metadata);
		const logPath = resolve(args.outDir, 'innovation-district-collector-run-log.json');
		await writeFile(metadataPath, JSON.stringify(metadata, null, 2), 'utf8');
		await writeFile(
			logPath,
			JSON.stringify(
				{
					schemaVersion: 'innovation-district-collector-log/v1',
					analysisId: ANALYSIS_ID,
					date: args.date,
					monthIndex: args.monthIndex,
					hours: args.hours,
					routeUrl: routeUrl(baseUrl),
					timingsMs: timings,
					diagnostics: {
						initial: initialDiagnostics,
						utciByHour: diagnosticsByHour,
						shadingIndex: shadingDiagnostics
					}
				},
				null,
				2
			),
			'utf8'
		);

		console.log(
			[
				`Innovation District GIS export complete`,
				`outDir=${args.outDir}`,
				`active=${metadata.activeCount}`,
				`canonical=${metadata.canonicalCount}`,
				`activeMask.source=${metadata.activeMask.source}`,
				`metadata=${metadataPath}`,
				`log=${logPath}`
			].join('\n')
		);
	} finally {
		await browser?.close().catch(() => undefined);
		if (server) await stopProcessTree(server);
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	main().catch((error) => {
		console.error(error instanceof Error ? error.stack : String(error));
		process.exit(1);
	});
}
