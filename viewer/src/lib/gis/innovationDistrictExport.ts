import { SURFACE_FLAGS, parseSurfaceFlags, type Analysis } from '$lib/types/analysis';

const EXPECTED_ACTIVE_MASK_SOURCE = 'base+road' as const;
const LITTLE_ENDIAN = 'little' as const;

type InnovationDistrictActiveMaskSource = typeof EXPECTED_ACTIVE_MASK_SOURCE;
type RawExportCoordinateSystem = 'projected-analysis';
type RawLayout = 'point-major' | 'point-major-xyz' | 'point-major-hour';
type RawArrayDtype = 'u32' | 'f32' | 'u8';
type RawExportFileKey =
	| 'canonicalIndices'
	| 'positions'
	| 'utci'
	| 'shadingIndex'
	| 'surfaceFlags';

type AnalysisActiveMaskSource = NonNullable<Analysis['metadata']['activeMask']>['source'];

export interface ActiveCellArrayBuildParams {
	activeCanonicalIndices?: Uint32Array;
	positions: Float32Array;
	utciByHour: readonly Float32Array[];
	shadingIndex: Float32Array;
	surfaceFlags: Uint8Array;
	hours: readonly number[];
	activeMaskSource?: AnalysisActiveMaskSource;
}

export interface RawExportFileDescriptorInput {
	fileName: string;
	checksum: string;
}

export interface RawExportMetadataBuildParams {
	schemaVersion: string;
	analysisId: string;
	sourceAnalysisId: string;
	sourceModelPath: string;
	sourceGeorefPath: string;
	declaredCrs: string;
	gridSize: number;
	coordinateSystem: RawExportCoordinateSystem;
	hours: readonly number[];
	canonicalIndices: Uint32Array;
	positions: Float32Array;
	utci: Float32Array;
	shadingIndex: Float32Array;
	activeMask: {
		source?: AnalysisActiveMaskSource;
		canonicalPointCount: number;
		checksum: string;
		signature: string;
	};
	surfaceFlags: Uint8Array;
	files?: Partial<Record<RawExportFileKey, RawExportFileDescriptorInput>>;
	timingsMs?: Record<string, number>;
}

export interface RawExportArrayDescriptor {
	dtype: RawArrayDtype;
	endianness: typeof LITTLE_ENDIAN;
	shape: number[];
	byteLength: number;
}

export interface RawExportMetadata {
	schemaVersion: string;
	analysisId: string;
	sourceAnalysisId: string;
	sourceModelPath: string;
	sourceGeorefPath: string;
	declaredCrs: string;
	gridSize: number;
	coordinateSystem: RawExportCoordinateSystem;
	canonicalCount: number;
	activeCount: number;
	hourCount: number;
	hours: number[];
	activeMask: {
		source: InnovationDistrictActiveMaskSource;
		checksum: string;
		signature: string;
	};
	layout: {
		canonicalIndices: RawLayout;
		positions: RawLayout;
		utci: RawLayout;
		shadingIndex: RawLayout;
		surfaceFlags: RawLayout;
	};
	arrays: {
		canonicalIndices: RawExportArrayDescriptor;
		positions: RawExportArrayDescriptor;
		utci: RawExportArrayDescriptor;
		shadingIndex: RawExportArrayDescriptor;
		surfaceFlags: RawExportArrayDescriptor;
	};
	files: Partial<Record<RawExportFileKey, RawExportFileDescriptorInput>>;
	timingsMs?: Record<string, number>;
}

export interface ActiveCellArraysResult {
	canonicalIndices: Uint32Array;
	positions: Float32Array;
	utci: Float32Array;
	shadingIndex: Float32Array;
	surfaceFlags: Uint8Array;
	layout: {
		canonicalIndices: RawLayout;
		positions: RawLayout;
		utci: RawLayout;
		shadingIndex: RawLayout;
		surfaceFlags: RawLayout;
	};
}

export interface ActiveCellSpatialArraysResult {
	canonicalIndices: Uint32Array;
	positions: Float32Array;
	activeMask: {
		source: InnovationDistrictActiveMaskSource;
		canonicalPointCount: number;
		activePointCount: number;
		inactivePointCount: number;
		activePointRatio: number;
		checksum: string;
		signature: string;
	};
	surfaceFlags: Uint8Array;
}

function assertTypedArrayPresent<T extends ArrayLike<number>>(
	value: T | undefined,
	name: string
): asserts value is T {
	if (value == null) {
		throw new Error(`${name} is required`);
	}
}

function assertExpectedActiveMaskSource(
	source: AnalysisActiveMaskSource | undefined,
	fieldName: string
): asserts source is InnovationDistrictActiveMaskSource {
	if (source == null) {
		throw new Error(`${fieldName} is required`);
	}
	if (source !== EXPECTED_ACTIVE_MASK_SOURCE) {
		throw new Error(
			`${fieldName} must be "${EXPECTED_ACTIVE_MASK_SOURCE}" for Innovation District export`
		);
	}
}

function assertExpectedCoordinateSystem(
	coordinateSystem: string,
	fieldName: string
): asserts coordinateSystem is RawExportCoordinateSystem {
	if (coordinateSystem !== 'projected-analysis') {
		throw new Error(`${fieldName} must be "projected-analysis" for Innovation District export`);
	}
}

function assertFiniteArrayValues(values: ArrayLike<number>, name: string): void {
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			throw new Error(`${name} contains non-finite value at index ${i}`);
		}
	}
}

function assertHours(hours: readonly number[]): void {
	for (let i = 0; i < hours.length; i++) {
		const hour = hours[i];
		if (!Number.isInteger(hour) || hour < 0) {
			throw new Error(`hours[${i}] must be a non-negative integer`);
		}
	}
}

function assertString(value: string | undefined, name: string): asserts value is string {
	if (typeof value !== 'string') {
		throw new Error(`${name} is required`);
	}
	if (value.trim().length === 0) {
		throw new Error(`${name} is required`);
	}
}

function assertPositiveFinite(value: number | undefined, name: string): asserts value is number {
	if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
		throw new Error(`${name} must be a positive finite number`);
	}
}

function assertPositionsLength(positions: Float32Array, activeCount: number): void {
	if (positions.length !== activeCount * 3) {
		throw new Error(
			`positions length ${positions.length} does not match active row count ${activeCount} * 3`
		);
	}
}

function assertUtciSlices(
	utciByHour: readonly Float32Array[],
	hours: readonly number[],
	activeCount: number
): void {
	if (utciByHour.length !== hours.length) {
		throw new Error(
			`utciByHour length ${utciByHour.length} does not match hour count ${hours.length}`
		);
	}
	for (let hourIndex = 0; hourIndex < utciByHour.length; hourIndex++) {
		const slice = utciByHour[hourIndex];
		if (slice.length !== activeCount) {
			throw new Error(
				`utciByHour[${hourIndex}] length ${slice.length} does not match active row count ${activeCount}`
			);
		}
		assertFiniteArrayValues(slice, `utciByHour[${hourIndex}]`);
	}
}

function assertPointMajorUtciLength(utci: Float32Array, activeCount: number, hourCount: number): void {
	const expectedLength = activeCount * hourCount;
	if (utci.length !== expectedLength) {
		throw new Error(`utci length ${utci.length} does not match activeCount * hourCount ${expectedLength}`);
	}
}

function assertShadingLength(shadingIndex: Float32Array, activeCount: number): void {
	if (shadingIndex.length !== activeCount) {
		throw new Error(
			`shadingIndex length ${shadingIndex.length} does not match active row count ${activeCount}`
		);
	}
}

function assertSurfaceFlagsArray(surfaceFlags: Uint8Array, activeCount: number): void {
	if (!(surfaceFlags instanceof Uint8Array)) {
		throw new Error('surfaceFlags must be a Uint8Array');
	}
	if (surfaceFlags.length !== activeCount) {
		throw new Error(
			`surfaceFlags length ${surfaceFlags.length} does not match active row count ${activeCount}`
		);
	}
}

function assertKnownSurfaceFlags(surfaceFlags: Uint8Array): void {
	const sampledSurfaceMask = SURFACE_FLAGS.ground | SURFACE_FLAGS.streetSurface;
	for (let index = 0; index < surfaceFlags.length; index += 1) {
		const value = surfaceFlags[index];
		parseSurfaceFlags(value);
		if ((value & sampledSurfaceMask) === 0) {
			throw new Error(
				`surfaceFlags[${index}] must include at least one sampled-surface bit (ground or streetSurface)`
			);
		}
	}
}

function assertByteLength(
	array: Uint32Array | Float32Array | Uint8Array,
	bytesPerElement: number,
	expectedElements: number,
	name: string
): void {
	if (array.byteLength !== bytesPerElement * expectedElements) {
		throw new Error(`${name} byteLength ${array.byteLength} does not match expected shape`);
	}
}

function buildArrayDescriptor(
	array: Uint32Array | Float32Array | Uint8Array,
	dtype: RawArrayDtype,
	shape: number[]
): RawExportArrayDescriptor {
	const expectedElements = shape.reduce((product, dimension) => product * dimension, 1);
	const bytesPerElement =
		dtype === 'u32'
			? Uint32Array.BYTES_PER_ELEMENT
			: dtype === 'u8'
				? Uint8Array.BYTES_PER_ELEMENT
				: Float32Array.BYTES_PER_ELEMENT;
	assertByteLength(array, bytesPerElement, expectedElements, `${dtype} array`);

	return {
		dtype,
		endianness: LITTLE_ENDIAN,
		shape: [...shape],
		byteLength: array.byteLength
	};
}

function cloneFiles(
	files: RawExportMetadataBuildParams['files']
): RawExportMetadata['files'] {
	const next: RawExportMetadata['files'] = {};
	if (files == null) return next;

	for (const [key, descriptor] of Object.entries(files) as Array<
		[
			keyof NonNullable<RawExportMetadataBuildParams['files']>,
			RawExportFileDescriptorInput | undefined
		]
	>) {
		if (descriptor == null) continue;
		if (
			key !== 'canonicalIndices' &&
			key !== 'positions' &&
			key !== 'utci' &&
			key !== 'shadingIndex' &&
			key !== 'surfaceFlags'
		) {
			throw new Error(`files.${String(key)} is not supported for Innovation District raw export`);
		}
		assertString(descriptor.fileName, `files.${key}.fileName`);
		assertString(descriptor.checksum, `files.${key}.checksum`);
		next[key] = {
			fileName: descriptor.fileName,
			checksum: descriptor.checksum
		};
	}

	return next;
}

function cloneTimings(timingsMs: RawExportMetadataBuildParams['timingsMs']): Record<string, number> | undefined {
	if (timingsMs == null) return undefined;

	const next: Record<string, number> = {};
	for (const [key, value] of Object.entries(timingsMs)) {
		if (!Number.isFinite(value) || value < 0) {
			throw new Error(`timingsMs.${key} must be a finite non-negative number`);
		}
		next[key] = value;
	}
	return next;
}

function validateRawArrayInputs(params: {
	canonicalIndices: Uint32Array;
	positions: Float32Array;
	utci: Float32Array;
	shadingIndex: Float32Array;
	surfaceFlags: Uint8Array;
	hours: readonly number[];
	activeMaskSource?: AnalysisActiveMaskSource;
}): { activeCount: number; hourCount: number } {
	const { canonicalIndices, positions, utci, shadingIndex, surfaceFlags, hours, activeMaskSource } =
		params;
	const activeCount = canonicalIndices.length;
	const hourCount = hours.length;
	const positionActiveCount = positions.length / 3;

	assertExpectedActiveMaskSource(activeMaskSource, 'activeMask.source');
	assertHours(hours);
	if (!Number.isInteger(positionActiveCount)) {
		throw new Error(`positions length ${positions.length} is not divisible by 3`);
	}
	if (positionActiveCount !== activeCount) {
		throw new Error(
			`canonicalIndices length ${activeCount} does not match positions active row count ${positionActiveCount}`
		);
	}
	assertPositionsLength(positions, activeCount);
	assertPointMajorUtciLength(utci, activeCount, hourCount);
	assertShadingLength(shadingIndex, activeCount);
	assertSurfaceFlagsArray(surfaceFlags, activeCount);
	assertFiniteArrayValues(canonicalIndices, 'canonicalIndices');
	assertFiniteArrayValues(positions, 'positions');
	assertFiniteArrayValues(utci, 'utci');
	assertFiniteArrayValues(shadingIndex, 'shadingIndex');
	assertKnownSurfaceFlags(surfaceFlags);

	return { activeCount, hourCount };
}

export function buildActiveCellSpatialArrays(analysis: Analysis): ActiveCellSpatialArraysResult {
	const activeMask = analysis.metadata.activeMask;
	if (activeMask == null) {
		throw new Error('metadata.activeMask is required for Innovation District export');
	}
	assertTypedArrayPresent(
		activeMask.activeCanonicalIndices,
		'metadata.activeMask.activeCanonicalIndices'
	);
	if (!('surfaceFlagsByActiveCell' in activeMask)) {
		throw new Error(
			'metadata.activeMask.surfaceFlagsByActiveCell is required for Innovation District classified export'
		);
	}
	assertExpectedActiveMaskSource(activeMask.source, 'metadata.activeMask.source');
	if (activeMask.inactivePointCount <= 0 || activeMask.activePointRatio >= 1) {
		throw new Error('Innovation District export refuses rectangular-only/canonical-only data');
	}
	if (activeMask.activeCanonicalIndices.length !== activeMask.activePointCount) {
		throw new Error('metadata.activeMask active row count does not match activeCanonicalIndices');
	}
	if (analysis.metadata.bounds == null) {
		throw new Error('metadata.bounds is required for Innovation District export');
	}
	assertPositiveFinite(analysis.metadata.grid_size, 'metadata.grid_size');
	assertString(activeMask.activeMaskChecksum, 'metadata.activeMask.activeMaskChecksum');
	assertString(activeMask.signature, 'metadata.activeMask.signature');

	const canonicalIndices = new Uint32Array(activeMask.activeCanonicalIndices);
	const surfaceFlags = new Uint8Array(activeMask.surfaceFlagsByActiveCell);
	if (analysis.data.numPositions !== activeMask.activePointCount) {
		throw new Error('analysis.data.numPositions does not match metadata.activeMask.activePointCount');
	}
	assertPositionsLength(analysis.data.positions, activeMask.activePointCount);
	assertFiniteArrayValues(analysis.data.positions, 'analysis.data.positions');
	assertSurfaceFlagsArray(surfaceFlags, activeMask.activePointCount);
	assertKnownSurfaceFlags(surfaceFlags);

	return {
		canonicalIndices,
		positions: new Float32Array(analysis.data.positions),
		activeMask: {
			source: EXPECTED_ACTIVE_MASK_SOURCE,
			canonicalPointCount: activeMask.canonicalPointCount,
			activePointCount: activeMask.activePointCount,
			inactivePointCount: activeMask.inactivePointCount,
			activePointRatio: activeMask.activePointRatio,
			checksum: activeMask.activeMaskChecksum,
			signature: activeMask.signature
		},
		surfaceFlags
	};
}

export function buildActiveCellArrays(params: ActiveCellArrayBuildParams): ActiveCellArraysResult {
	const {
		activeCanonicalIndices,
		positions,
		utciByHour,
		shadingIndex,
		surfaceFlags,
		hours,
		activeMaskSource
	} = params;

	assertTypedArrayPresent(activeCanonicalIndices, 'activeCanonicalIndices');
	assertExpectedActiveMaskSource(activeMaskSource, 'activeMaskSource');
	assertHours(hours);

	const activeCount = activeCanonicalIndices.length;
	assertPositionsLength(positions, activeCount);
	assertUtciSlices(utciByHour, hours, activeCount);
	assertShadingLength(shadingIndex, activeCount);
	assertSurfaceFlagsArray(surfaceFlags, activeCount);
	assertFiniteArrayValues(activeCanonicalIndices, 'activeCanonicalIndices');
	assertFiniteArrayValues(positions, 'positions');
	assertFiniteArrayValues(shadingIndex, 'shadingIndex');
	assertKnownSurfaceFlags(surfaceFlags);

	const canonicalIndices = new Uint32Array(activeCanonicalIndices);
	const nextPositions = new Float32Array(positions);
	const nextShadingIndex = new Float32Array(shadingIndex);
	const nextSurfaceFlags = new Uint8Array(surfaceFlags);
	const utci = new Float32Array(activeCount * hours.length);

	let offset = 0;
	for (let pointIndex = 0; pointIndex < activeCount; pointIndex++) {
		for (let hourIndex = 0; hourIndex < utciByHour.length; hourIndex++) {
			utci[offset++] = utciByHour[hourIndex][pointIndex];
		}
	}

	return {
		canonicalIndices,
		positions: nextPositions,
		utci,
		shadingIndex: nextShadingIndex,
		surfaceFlags: nextSurfaceFlags,
		layout: {
			canonicalIndices: 'point-major',
			positions: 'point-major-xyz',
			utci: 'point-major-hour',
			shadingIndex: 'point-major',
			surfaceFlags: 'point-major'
		}
	};
}

export function buildRawExportMetadata(params: RawExportMetadataBuildParams): RawExportMetadata {
	const {
		schemaVersion,
		analysisId,
		sourceAnalysisId,
		sourceModelPath,
		sourceGeorefPath,
		declaredCrs,
		gridSize,
		coordinateSystem,
		hours,
		canonicalIndices,
		positions,
		utci,
		shadingIndex,
		surfaceFlags,
		activeMask,
		files,
		timingsMs
	} = params;

	assertString(schemaVersion, 'schemaVersion');
	assertString(analysisId, 'analysisId');
	assertString(sourceAnalysisId, 'sourceAnalysisId');
	assertString(sourceModelPath, 'sourceModelPath');
	assertString(sourceGeorefPath, 'sourceGeorefPath');
	assertString(declaredCrs, 'declaredCrs');
	assertPositiveFinite(gridSize, 'gridSize');
	assertExpectedCoordinateSystem(coordinateSystem, 'coordinateSystem');
	assertString(activeMask.checksum, 'activeMask.checksum');
	assertString(activeMask.signature, 'activeMask.signature');

	const { activeCount, hourCount } = validateRawArrayInputs({
		canonicalIndices,
		positions,
		utci,
		shadingIndex,
		surfaceFlags,
		hours,
		activeMaskSource: activeMask.source
	});

	return {
		schemaVersion,
		analysisId,
		sourceAnalysisId,
		sourceModelPath,
		sourceGeorefPath,
		declaredCrs,
		gridSize,
		coordinateSystem,
		canonicalCount: activeMask.canonicalPointCount,
		activeCount,
		hourCount,
		hours: [...hours],
		activeMask: {
			source: EXPECTED_ACTIVE_MASK_SOURCE,
			checksum: activeMask.checksum,
			signature: activeMask.signature
		},
		layout: {
			canonicalIndices: 'point-major',
			positions: 'point-major-xyz',
			utci: 'point-major-hour',
			shadingIndex: 'point-major',
			surfaceFlags: 'point-major'
		},
		arrays: {
			canonicalIndices: buildArrayDescriptor(canonicalIndices, 'u32', [activeCount]),
			positions: buildArrayDescriptor(positions, 'f32', [activeCount, 3]),
			utci: buildArrayDescriptor(utci, 'f32', [activeCount, hourCount]),
			shadingIndex: buildArrayDescriptor(shadingIndex, 'f32', [activeCount]),
			surfaceFlags: buildArrayDescriptor(surfaceFlags, 'u8', [activeCount])
		},
		files: cloneFiles(files),
		timingsMs: cloneTimings(timingsMs)
	};
}

// TypeScript ownership boundary: raw arrays + raw metadata only.
// GeoJSON, summaries, and final GIS post-processing belong to Python.
