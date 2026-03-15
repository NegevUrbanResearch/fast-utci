export type MrtTermName = 'short_erf' | 'long_erf' | 'short_dmrt' | 'long_dmrt' | 'solar' | 'sky';

export interface TermSeries {
	ref: Float32Array | readonly number[];
	webgpu: readonly number[];
}

export type OptionalTermSeries = Partial<Record<MrtTermName, TermSeries>>;

export interface WorstCellRow {
	index: number;
	hour: number;
	pointIndex: number;
	ref: number;
	webgpu: number;
	diff: number;
	absDiff: number;
	termDeltas: Partial<Record<MrtTermName, number>>;
}

function validatePairLength(ref: Float32Array | readonly number[], webgpu: readonly number[], label: string): void {
	if (ref.length !== webgpu.length) {
		throw new Error(`${label} length mismatch: ref=${ref.length}, webgpu=${webgpu.length}`);
	}
}

export function flatIndexToHourPoint(index: number, numHours: number): { hour: number; pointIndex: number } {
	if (!Number.isInteger(index) || index < 0) {
		throw new Error(`index must be a non-negative integer, got ${index}`);
	}
	if (!Number.isInteger(numHours) || numHours <= 0) {
		throw new Error(`numHours must be a positive integer, got ${numHours}`);
	}
	return {
		hour: index % numHours,
		pointIndex: Math.floor(index / numHours)
	};
}

function collectIndices(length: number, explicit?: readonly number[]): number[] {
	if (!explicit || explicit.length === 0) {
		return Array.from({ length }, (_, index) => index);
	}
	const deduped = new Set<number>();
	for (const value of explicit) {
		if (!Number.isInteger(value) || value < 0 || value >= length) continue;
		deduped.add(value);
	}
	return Array.from(deduped);
}

export function extractTopMrtDeltas(args: {
	refMrt: Float32Array | readonly number[];
	webgpuMrt: readonly number[];
	numPositions: number;
	topN: number;
	indices?: readonly number[];
	terms?: OptionalTermSeries;
}): WorstCellRow[] {
	const { refMrt, webgpuMrt, numPositions, topN, indices, terms } = args;
	validatePairLength(refMrt, webgpuMrt, 'mrt');
	if (!Number.isInteger(topN) || topN <= 0) {
		throw new Error(`topN must be a positive integer, got ${topN}`);
	}
	if (!Number.isInteger(numPositions) || numPositions <= 0) {
		throw new Error(`numPositions must be a positive integer, got ${numPositions}`);
	}
	if (refMrt.length % numPositions !== 0) {
		throw new Error(`mrt length ${refMrt.length} is not divisible by numPositions ${numPositions}`);
	}
	const numHours = refMrt.length / numPositions;

	for (const [name, series] of Object.entries(terms ?? {})) {
		if (!series) continue;
		validatePairLength(series.ref, series.webgpu, name);
		if (series.ref.length !== refMrt.length) {
			throw new Error(`${name} length mismatch vs mrt: ${series.ref.length} !== ${refMrt.length}`);
		}
	}

	const candidates = collectIndices(refMrt.length, indices);
	const rows = candidates.map((index) => {
		const ref = refMrt[index];
		const webgpu = webgpuMrt[index];
		const diff = webgpu - ref;
		const termDeltas: Partial<Record<MrtTermName, number>> = {};

		for (const [name, series] of Object.entries(terms ?? {}) as Array<[MrtTermName, TermSeries | undefined]>) {
			if (!series) continue;
			termDeltas[name] = series.webgpu[index] - series.ref[index];
		}

		return {
			index,
			...flatIndexToHourPoint(index, numHours),
			ref,
			webgpu,
			diff,
			absDiff: Math.abs(diff),
			termDeltas
		} satisfies WorstCellRow;
	});

	rows.sort((a, b) => b.absDiff - a.absDiff);
	return rows.slice(0, Math.min(topN, rows.length));
}
