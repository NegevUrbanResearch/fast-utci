export type ShadingIndexPosition = { x: number; y: number; z: number };

export type ShadingIndexWorstCell = {
	pointIndex: number;
	position?: ShadingIndexPosition;
	pythonValue: number;
	webgpuValue: number;
	absoluteError: number;
	solarBitMismatchCount: number | null;
	attributedToSolarBitFlip: boolean;
};

export type CompareShadingIndexResult = {
	pass: boolean;
	strictPass: boolean;
	numPoints: number;
	tolerance: number;
	maxAbsoluteError: number;
	meanAbsoluteError: number;
	mismatchCountAboveTolerance: number;
	nonFinitePythonValueCount: number;
	nonFiniteWebgpuValueCount: number;
	solarBitFlipAttributedMismatchCount: number;
	worstCells: ShadingIndexWorstCell[];
	caveats: string[];
};

function valueAt(values: ArrayLike<number>, index: number): number {
	return values[index];
}

function isFiniteNumber(value: number): boolean {
	return Number.isFinite(value);
}

function positionAt(positions: ArrayLike<number> | undefined, pointIndex: number): ShadingIndexPosition | undefined {
	if (!positions) return undefined;
	const base = pointIndex * 3;
	if (base + 2 >= positions.length) return undefined;
	return {
		x: valueAt(positions, base),
		y: valueAt(positions, base + 1),
		z: valueAt(positions, base + 2)
	};
}

function isSolarBitFlipAttributable(params: {
	absoluteError: number;
	tolerance: number;
	sunUpCount?: number;
	solarBitMismatchCount: number | null;
}): boolean {
	if (!params.sunUpCount || params.sunUpCount <= 0 || params.solarBitMismatchCount == null) {
		return false;
	}
	if (params.solarBitMismatchCount <= 0) return false;
	const maxExpectedErrorFromFlips = params.solarBitMismatchCount / params.sunUpCount;
	return params.absoluteError <= maxExpectedErrorFromFlips + params.tolerance;
}

export function compareShadingIndex(params: {
	python: ArrayLike<number>;
	webgpu: ArrayLike<number>;
	positions?: ArrayLike<number>;
	tolerance?: number;
	maxWorstCells?: number;
	sunUpCount?: number;
	solarBitMismatchCounts?: ArrayLike<number>;
}): CompareShadingIndexResult {
	const {
		python,
		webgpu,
		positions,
		tolerance = 1e-6,
		maxWorstCells = 10,
		sunUpCount,
		solarBitMismatchCounts
	} = params;

	if (python.length !== webgpu.length) {
		throw new Error(`Shading Index length mismatch: python ${python.length} vs webgpu ${webgpu.length}`);
	}
	if (positions && positions.length < python.length * 3) {
		throw new Error(
			`Shading Index positions length mismatch: expected at least ${python.length * 3}, got ${positions.length}`
		);
	}
	if (solarBitMismatchCounts && solarBitMismatchCounts.length !== python.length) {
		throw new Error(
			`Shading Index solar bit mismatch count length mismatch: expected ${python.length}, got ${solarBitMismatchCounts.length}`
		);
	}

	let maxAbsoluteError = 0;
	let sumAbsoluteError = 0;
	let mismatchCountAboveTolerance = 0;
	let nonFinitePythonValueCount = 0;
	let nonFiniteWebgpuValueCount = 0;
	let solarBitFlipAttributedMismatchCount = 0;
	const worstCells: ShadingIndexWorstCell[] = [];

	for (let pointIndex = 0; pointIndex < python.length; pointIndex += 1) {
		const pythonValue = valueAt(python, pointIndex);
		const webgpuValue = valueAt(webgpu, pointIndex);
		const pythonValueIsFinite = isFiniteNumber(pythonValue);
		const webgpuValueIsFinite = isFiniteNumber(webgpuValue);
		if (!pythonValueIsFinite) nonFinitePythonValueCount += 1;
		if (!webgpuValueIsFinite) nonFiniteWebgpuValueCount += 1;
		const absoluteError =
			pythonValueIsFinite && webgpuValueIsFinite
				? Math.abs(webgpuValue - pythonValue)
				: Number.POSITIVE_INFINITY;
		const solarBitMismatchCount = solarBitMismatchCounts
			? valueAt(solarBitMismatchCounts, pointIndex)
			: null;
		const attributedToSolarBitFlip =
			pythonValueIsFinite &&
			webgpuValueIsFinite &&
			absoluteError > tolerance &&
			isSolarBitFlipAttributable({
				absoluteError,
				tolerance,
				sunUpCount,
				solarBitMismatchCount
			});

		sumAbsoluteError += absoluteError;
		if (absoluteError > maxAbsoluteError) maxAbsoluteError = absoluteError;
		if (absoluteError > tolerance) {
			mismatchCountAboveTolerance += 1;
			if (attributedToSolarBitFlip) solarBitFlipAttributedMismatchCount += 1;
		}

		const cell: ShadingIndexWorstCell = {
			pointIndex,
			position: positionAt(positions, pointIndex),
			pythonValue,
			webgpuValue,
			absoluteError,
			solarBitMismatchCount,
			attributedToSolarBitFlip
		};
		worstCells.push(cell);
	}

	worstCells.sort((a, b) => b.absoluteError - a.absoluteError || a.pointIndex - b.pointIndex);
	const trimmedWorstCells = worstCells.slice(0, Math.max(0, maxWorstCells));
	const strictPass = mismatchCountAboveTolerance === 0;
	const pass =
		strictPass || mismatchCountAboveTolerance === solarBitFlipAttributedMismatchCount;
	const caveats =
		!strictPass && pass
			? [
					`${solarBitFlipAttributedMismatchCount} shading-index mismatch(es) above tolerance are attributable to known solar ray bit flips.`
				]
			: [];

	return {
		pass,
		strictPass,
		numPoints: python.length,
		tolerance,
		maxAbsoluteError,
		meanAbsoluteError: python.length > 0 ? sumAbsoluteError / python.length : 0,
		mismatchCountAboveTolerance,
		nonFinitePythonValueCount,
		nonFiniteWebgpuValueCount,
		solarBitFlipAttributedMismatchCount,
		worstCells: trimmedWorstCells,
		caveats
	};
}
