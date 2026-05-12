/**
 * Load reference intermediate-stage data (solar or sky exposure) from the filesystem.
 * Node only; use dynamic import of fs so the module can load in browser (throws if called there).
 */

export interface SolarReference {
	numPositions: number;
	numHours: number;
	solarExposure: Float32Array;
}

export interface SkyReference {
	numPositions: number;
	skyExposure: Float32Array;
}

export interface MrtReference {
	numPositions: number;
	numHours: number;
	mrt: Float32Array;
	short_erf?: Float32Array;
	long_erf?: Float32Array;
	short_dmrt?: Float32Array;
	long_dmrt?: Float32Array;
}

/**
 * Load reference intermediates from basePath_stage.json. Node only.
 * @param basePath - Full path without extension, e.g. .../Ben-Gurion/20250815_grid_2m_fullday
 * @param stage - 'solar' | 'sky' | 'mrt'
 */
export function loadReferenceIntermediatesFromFs(basePath: string, stage: 'solar'): Promise<SolarReference>;
export function loadReferenceIntermediatesFromFs(basePath: string, stage: 'sky'): Promise<SkyReference>;
export function loadReferenceIntermediatesFromFs(basePath: string, stage: 'mrt'): Promise<MrtReference>;
export async function loadReferenceIntermediatesFromFs(
	basePath: string,
	stage: 'solar' | 'sky' | 'mrt'
): Promise<SolarReference | SkyReference | MrtReference> {
	const { readFileSync } = await import('node:fs');
	const path =
		stage === 'solar' ? `${basePath}_solar.json` : stage === 'sky' ? `${basePath}_sky.json` : `${basePath}_mrt.json`;
	let raw: string;
	try {
		raw = readFileSync(path, 'utf8');
	} catch (e) {
		throw new Error(`Failed to load ${stage} reference from ${path}: ${e}`);
	}
	const data = JSON.parse(raw) as unknown;

	if (stage === 'solar') {
		if (
			typeof data !== 'object' ||
			data === null ||
			typeof (data as Record<string, unknown>).numPositions !== 'number' ||
			typeof (data as Record<string, unknown>).numHours !== 'number' ||
			!(data as Record<string, unknown>).solarExposure ||
			!Array.isArray((data as Record<string, unknown>).solarExposure)
		) {
			throw new Error(`Invalid solar reference shape in ${path}: expected { numPositions, numHours, solarExposure: number[] }`);
		}
		const { numPositions, numHours, solarExposure: arr } = data as {
			numPositions: number;
			numHours: number;
			solarExposure: number[];
		};
		const expected = numPositions * numHours;
		if (arr.length !== expected) {
			throw new Error(
				`Invalid solar reference in ${path}: solarExposure.length ${arr.length} !== numPositions * numHours (${expected})`
			);
		}
		return {
			numPositions,
			numHours,
			solarExposure: new Float32Array(arr)
		};
	}

	if (stage === 'mrt') {
		if (
			typeof data !== 'object' ||
			data === null ||
			typeof (data as Record<string, unknown>).numPositions !== 'number' ||
			typeof (data as Record<string, unknown>).numHours !== 'number' ||
			!(data as Record<string, unknown>).mrt ||
			!Array.isArray((data as Record<string, unknown>).mrt)
		) {
			throw new Error(`Invalid MRT reference shape in ${path}: expected { numPositions, numHours, mrt: number[] }`);
		}
		const { numPositions, numHours, mrt: arr } = data as {
			numPositions: number;
			numHours: number;
			mrt: number[];
		};
		const expected = numPositions * numHours;
		if (arr.length !== expected) {
			throw new Error(
				`Invalid MRT reference in ${path}: mrt.length ${arr.length} !== numPositions * numHours (${expected})`
			);
		}
		const out: MrtReference = {
			numPositions,
			numHours,
			mrt: new Float32Array(arr)
		};
		const rawObj = data as Record<string, unknown>;
		if (Object.prototype.hasOwnProperty.call(rawObj, 'short_erf')) {
			if (!Array.isArray(rawObj.short_erf)) {
				throw new Error(`Invalid MRT reference in ${path}: short_erf must be number[] when present`);
			}
			if (rawObj.short_erf.length !== expected) {
				throw new Error(
					`Invalid MRT reference in ${path}: short_erf.length ${rawObj.short_erf.length} !== numPositions * numHours (${expected})`
				);
			}
			out.short_erf = new Float32Array(rawObj.short_erf as number[]);
		}
		if (Object.prototype.hasOwnProperty.call(rawObj, 'long_erf')) {
			if (!Array.isArray(rawObj.long_erf)) {
				throw new Error(`Invalid MRT reference in ${path}: long_erf must be number[] when present`);
			}
			if (rawObj.long_erf.length !== expected) {
				throw new Error(
					`Invalid MRT reference in ${path}: long_erf.length ${rawObj.long_erf.length} !== numPositions * numHours (${expected})`
				);
			}
			out.long_erf = new Float32Array(rawObj.long_erf as number[]);
		}
		if (Object.prototype.hasOwnProperty.call(rawObj, 'short_dmrt')) {
			if (!Array.isArray(rawObj.short_dmrt)) {
				throw new Error(`Invalid MRT reference in ${path}: short_dmrt must be number[] when present`);
			}
			if (rawObj.short_dmrt.length !== expected) {
				throw new Error(
					`Invalid MRT reference in ${path}: short_dmrt.length ${rawObj.short_dmrt.length} !== numPositions * numHours (${expected})`
				);
			}
			out.short_dmrt = new Float32Array(rawObj.short_dmrt as number[]);
		}
		if (Object.prototype.hasOwnProperty.call(rawObj, 'long_dmrt')) {
			if (!Array.isArray(rawObj.long_dmrt)) {
				throw new Error(`Invalid MRT reference in ${path}: long_dmrt must be number[] when present`);
			}
			if (rawObj.long_dmrt.length !== expected) {
				throw new Error(
					`Invalid MRT reference in ${path}: long_dmrt.length ${rawObj.long_dmrt.length} !== numPositions * numHours (${expected})`
				);
			}
			out.long_dmrt = new Float32Array(rawObj.long_dmrt as number[]);
		}
		return out;
	}

	// stage === 'sky'
	if (
		typeof data !== 'object' ||
		data === null ||
		typeof (data as Record<string, unknown>).numPositions !== 'number' ||
		!(data as Record<string, unknown>).skyExposure ||
		!Array.isArray((data as Record<string, unknown>).skyExposure)
	) {
		throw new Error(`Invalid sky reference shape in ${path}: expected { numPositions, skyExposure: number[] }`);
	}
	const { numPositions, skyExposure: arr } = data as { numPositions: number; skyExposure: number[] };
	if (arr.length !== numPositions) {
		throw new Error(`Invalid sky reference in ${path}: skyExposure.length ${arr.length} !== numPositions (${numPositions})`);
	}
	return {
		numPositions,
		skyExposure: new Float32Array(arr)
	};
}
