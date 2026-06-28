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

	const mustNumberArray = (v: unknown, label: string): number[] => {
		if (!Array.isArray(v) || !v.every((x) => typeof x === 'number')) {
			throw new Error(`Invalid WebGPU artifact ${label}: expected number[]`);
		}
		return v as number[];
	};

	const hasOwn = (obj: Record<string, unknown>, key: string): boolean => Object.prototype.hasOwnProperty.call(obj, key);

	const readOptional = (suffix: string): unknown => {
		try {
			const raw = readFileSync(`${basePath}${suffix}`, 'utf8');
			return JSON.parse(raw) as unknown;
		} catch (error) {
			if (error instanceof Error && 'code' in error && (error as { code?: string }).code === 'ENOENT') {
				return undefined;
			}
			throw new Error(`Failed to parse ${basePath}${suffix}: ${error instanceof Error ? error.message : String(error)}`);
		}
	};

	const validateLength = (arr: number[], expected: number, label: string): void => {
		if (arr.length !== expected) {
			throw new Error(`Invalid WebGPU artifact ${label}: length ${arr.length} !== expected ${expected}`);
		}
	};

	const mustRecord = (v: unknown, label: string): Record<string, unknown> => {
		if (typeof v !== 'object' || v === null) {
			throw new Error(`Invalid WebGPU artifact ${label}: expected object`);
		}
		return v as Record<string, unknown>;
	};

	const mustNumber = (v: unknown, label: string): number => {
		if (typeof v !== 'number') {
			throw new Error(`Invalid WebGPU artifact ${label}: expected number`);
		}
		return v;
	};

	const validateOptionalPointwise = (
		obj: Record<string, unknown>,
		key: 'short_erf' | 'long_erf' | 'short_dmrt' | 'long_dmrt',
		expected: number
	): number[] | undefined => {
		if (!hasOwn(obj, key)) {
			return undefined;
		}
		const arr = mustNumberArray(obj[key], key);
		validateLength(arr, expected, key);
		return arr;
	};

	const solarRaw = readOptional('_webgpu_solar.json');
	if (solarRaw != null) {
		const obj = mustRecord(solarRaw, '_webgpu_solar.json');
		const numPositions = mustNumber(obj.numPositions, 'solar.numPositions');
		const numHours = mustNumber(obj.numHours, 'solar.numHours');
		const solarExposure = mustNumberArray(obj.solarExposure, 'solar.solarExposure');
		validateLength(solarExposure, numPositions * numHours, 'solar.solarExposure');
		result.solar = { numPositions, numHours, solarExposure };
	}

	const skyRaw = readOptional('_webgpu_sky.json');
	if (skyRaw != null) {
		const obj = mustRecord(skyRaw, '_webgpu_sky.json');
		const numPositions = mustNumber(obj.numPositions, 'sky.numPositions');
		const skyExposure = mustNumberArray(obj.skyExposure, 'sky.skyExposure');
		validateLength(skyExposure, numPositions, 'sky.skyExposure');
		result.sky = { numPositions, skyExposure };
	}

	const mrtRaw = readOptional('_webgpu_mrt.json');
	if (mrtRaw != null) {
		const obj = mustRecord(mrtRaw, '_webgpu_mrt.json');
		const numPositions = mustNumber(obj.numPositions, 'mrt.numPositions');
		const numHours = mustNumber(obj.numHours, 'mrt.numHours');
		const expected = numPositions * numHours;
		const mrt = mustNumberArray(obj.mrt, 'mrt.mrt');
		validateLength(mrt, expected, 'mrt.mrt');
		const short_erf = validateOptionalPointwise(obj, 'short_erf', expected);
		const long_erf = validateOptionalPointwise(obj, 'long_erf', expected);
		const short_dmrt = validateOptionalPointwise(obj, 'short_dmrt', expected);
		const long_dmrt = validateOptionalPointwise(obj, 'long_dmrt', expected);

		result.mrt = {
			numPositions,
			numHours,
			mrt,
			...(short_erf ? { short_erf } : {}),
			...(long_erf ? { long_erf } : {}),
			...(short_dmrt ? { short_dmrt } : {}),
			...(long_dmrt ? { long_dmrt } : {})
		};
	}

	const utciRaw = readOptional('_webgpu_utci.json');
	if (utciRaw != null) {
		const obj = mustRecord(utciRaw, '_webgpu_utci.json');
		const numPoints = mustNumber(obj.numPoints, 'utci.numPoints');
		const numHours = mustNumber(obj.numHours, 'utci.numHours');
		const utciByHourRaw = obj.utciByHour;
		if (!Array.isArray(utciByHourRaw)) {
			throw new Error('Invalid WebGPU artifact utci.utciByHour: expected number[][]');
		}
		if (utciByHourRaw.length !== numHours) {
			throw new Error(
				`Invalid WebGPU artifact utci.utciByHour: hour count ${utciByHourRaw.length} !== numHours ${numHours}`
			);
		}
		const utciByHour = utciByHourRaw.map((hour, hourIdx) => {
			const points = mustNumberArray(hour, `utci.utciByHour[${hourIdx}]`);
			validateLength(points, numPoints, `utci.utciByHour[${hourIdx}]`);
			return points;
		});
		let positions: number[] | undefined;
		if (hasOwn(obj, 'positions')) {
			positions = mustNumberArray(obj.positions, 'utci.positions');
			validateLength(positions, numPoints * 3, 'utci.positions');
		}
		const rangeObj = mustRecord(obj.utci_range, 'utci.utci_range');
		const utci_range = {
			min: mustNumber(rangeObj.min, 'utci.utci_range.min'),
			max: mustNumber(rangeObj.max, 'utci.utci_range.max'),
			mean: mustNumber(rangeObj.mean, 'utci.utci_range.mean')
		};
		result.utci = { numPoints, numHours, ...(positions ? { positions } : {}), utciByHour, utci_range };
	}

	return result;
}
