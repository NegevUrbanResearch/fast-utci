/**
 * Load WebGPU-collected parity files from disk ({basePath}_webgpu_*.json).
 * Node only; missing file → that key omitted (no throw).
 */

export interface WebgpuSolar {
	numPositions: number;
	numHours: number;
	solarExposure: number[];
}

export interface WebgpuSky {
	numPositions: number;
	skyExposure: number[];
}

export interface WebgpuMrt {
	numPositions: number;
	numHours: number;
	mrt: number[];
	short_erf?: number[];
	long_erf?: number[];
	short_dmrt?: number[];
	long_dmrt?: number[];
}

export interface WebgpuUtci {
	numPoints: number;
	numHours: number;
	positions?: number[];
	utciByHour: number[][];
	utci_range: { min: number; max: number; mean: number };
}

export interface LoadWebgpuCollectedResult {
	solar?: WebgpuSolar;
	sky?: WebgpuSky;
	mrt?: WebgpuMrt;
	utci?: WebgpuUtci;
}

export async function loadWebgpuCollectedFromFs(basePath: string): Promise<LoadWebgpuCollectedResult> {
	const { readFileSync } = await import('node:fs');
	const result: LoadWebgpuCollectedResult = {};

	const readOptional = (suffix: string): unknown => {
		try {
			const raw = readFileSync(`${basePath}${suffix}`, 'utf8');
			return JSON.parse(raw) as unknown;
		} catch {
			return undefined;
		}
	};

	const solarRaw = readOptional('_webgpu_solar.json');
	if (
		solarRaw &&
		typeof solarRaw === 'object' &&
		solarRaw !== null &&
		typeof (solarRaw as Record<string, unknown>).numPositions === 'number' &&
		typeof (solarRaw as Record<string, unknown>).numHours === 'number' &&
		Array.isArray((solarRaw as Record<string, unknown>).solarExposure)
	) {
		result.solar = solarRaw as WebgpuSolar;
	}

	const skyRaw = readOptional('_webgpu_sky.json');
	if (
		skyRaw &&
		typeof skyRaw === 'object' &&
		skyRaw !== null &&
		typeof (skyRaw as Record<string, unknown>).numPositions === 'number' &&
		Array.isArray((skyRaw as Record<string, unknown>).skyExposure)
	) {
		result.sky = skyRaw as WebgpuSky;
	}

	const mrtRaw = readOptional('_webgpu_mrt.json');
	if (
		mrtRaw &&
		typeof mrtRaw === 'object' &&
		mrtRaw !== null &&
		typeof (mrtRaw as Record<string, unknown>).numPositions === 'number' &&
		typeof (mrtRaw as Record<string, unknown>).numHours === 'number' &&
		Array.isArray((mrtRaw as Record<string, unknown>).mrt)
	) {
		result.mrt = mrtRaw as WebgpuMrt;
	}

	const utciRaw = readOptional('_webgpu_utci.json');
	if (
		utciRaw &&
		typeof utciRaw === 'object' &&
		utciRaw !== null &&
		typeof (utciRaw as Record<string, unknown>).numPoints === 'number' &&
		typeof (utciRaw as Record<string, unknown>).numHours === 'number' &&
		Array.isArray((utciRaw as Record<string, unknown>).utciByHour) &&
		typeof (utciRaw as Record<string, unknown>).utci_range === 'object'
	) {
		result.utci = utciRaw as WebgpuUtci;
	}

	return result;
}
