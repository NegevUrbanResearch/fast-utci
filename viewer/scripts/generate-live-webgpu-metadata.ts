import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { canonicalGridPoints } from '../src/lib/compute/core/canonicalGrid';

export interface AnalysisBounds {
	x_min: number;
	x_max: number;
	y_min: number;
	y_max: number;
	z?: number;
}

export interface LayerBounds {
	minX: number;
	maxX: number;
	minY: number;
	maxY: number;
	minZ: number;
	maxZ: number;
}

export interface LayerInventoryEntry {
	name: string;
	bounds: LayerBounds | null;
	primitiveCount: number;
	surfacePrimitiveCount: number;
	linePrimitiveCount: number;
	pointPrimitiveCount: number;
}

export interface GlbLayerInventory {
	layers: LayerInventoryEntry[];
}

export interface LayerRoles {
	samplingLayers: string[];
	occluderLayers: string[];
	ignoredLayers: string[];
	unknownLayers: string[];
}

type CoordinateSystem = 'xy_ground' | 'xz_ground';

interface CliOptions {
	model: string;
	out: string;
	analysisId: string;
	projectId: string;
	gridSize: number;
	date: string;
	coordinateSystem: CoordinateSystem;
	sampleHeight: number;
	weatherProfile: string;
	samplingLayers?: string[];
	occluderLayers?: string[];
	ignoredLayers?: string[];
}

interface GlbJson {
	scene?: number;
	scenes?: Array<{ nodes?: number[] }>;
	nodes?: GlbNode[];
	meshes?: GlbMesh[];
	accessors?: GlbAccessor[];
	bufferViews?: GlbBufferView[];
	buffers?: Array<{ byteLength: number }>;
}

interface GlbNode {
	name?: string;
	mesh?: number;
	children?: number[];
	matrix?: number[];
	translation?: number[];
	rotation?: number[];
	scale?: number[];
}

interface GlbMesh {
	primitives?: GlbPrimitive[];
}

interface GlbPrimitive {
	attributes?: Record<string, number>;
	mode?: number;
}

interface GlbAccessor {
	bufferView?: number;
	byteOffset?: number;
	componentType: number;
	normalized?: boolean;
	count: number;
	type: string;
	min?: number[];
	max?: number[];
	sparse?: unknown;
}

interface GlbBufferView {
	buffer: number;
	byteOffset?: number;
	byteLength: number;
	byteStride?: number;
}

interface ParsedGlb {
	json: GlbJson;
	binChunk: Buffer;
}

const TRIANGLE_MODES = new Set([4, 5, 6]);
const LINE_MODES = new Set([1, 2, 3]);
const POINT_MODE = 0;

const SAMPLING_LAYER_NAMES = new Set([
	'ground',
	'terrain',
	'street',
	'road',
	'roads',
	'sidewalk',
	'sidewalks',
	'parking',
	'walkway',
	'train_tracks',
	'train_track'
]);

const OCCLUDER_LAYER_NAMES = new Set([
	'existing_buildings',
	'existing_building',
	'buildings',
	'building',
	'new_buildings',
	'new_building',
	'trees_canopy',
	'tree_canopy',
	'trees',
	'tree',
	'vegetation',
	'new_trees',
	'new_tree'
]);

const IGNORED_LAYER_NAMES = new Set([
	'district_outline',
	'outline',
	'trees_point',
	'tree_point'
]);

const COMPONENT_BYTE_SIZE: Record<number, number> = {
	5120: 1,
	5121: 1,
	5122: 2,
	5123: 2,
	5125: 4,
	5126: 4
};

const TYPE_COMPONENTS: Record<string, number> = {
	SCALAR: 1,
	VEC2: 2,
	VEC3: 3,
	VEC4: 4,
	MAT2: 4,
	MAT3: 9,
	MAT4: 16
};

const BEER_SHEVA_WEATHER = {
	epw_file: 'data/weather/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw',
	location: {
		latitude: 31.2515,
		longitude: 34.7995,
		timezone: 2.0,
		city: 'Beer.Sheva'
	}
} as const;

export async function readGlbLayerInventory(modelPath: string): Promise<GlbLayerInventory> {
	const parsed = parseGlb(await readFile(modelPath));
	return buildLayerInventory(parsed);
}

export function inferLayerRoles(layers: readonly LayerInventoryEntry[], overrides: Partial<LayerRoles> = {}): LayerRoles {
	const samplingOverride = overrides.samplingLayers ? new Set(overrides.samplingLayers) : null;
	const occluderOverride = overrides.occluderLayers ? new Set(overrides.occluderLayers) : null;
	const ignoredOverride = overrides.ignoredLayers ? new Set(overrides.ignoredLayers) : null;

	const samplingLayers: string[] = [];
	const occluderLayers: string[] = [];
	const ignoredLayers: string[] = [];
	const unknownLayers: string[] = [];

	for (const layer of layers) {
		const normalized = normalizeLayerName(layer.name);
		const hasSurfaceGeometry = layer.surfacePrimitiveCount > 0;
		const isSamplingByDefault = SAMPLING_LAYER_NAMES.has(normalized) && hasSurfaceGeometry;
		const isOccluderByDefault = OCCLUDER_LAYER_NAMES.has(normalized) && hasSurfaceGeometry;
		const isIgnoredByDefault = IGNORED_LAYER_NAMES.has(normalized);

		if (ignoredOverride?.has(layer.name)) {
			ignoredLayers.push(layer.name);
		} else if (samplingOverride?.has(layer.name)) {
			samplingLayers.push(layer.name);
		} else if (occluderOverride?.has(layer.name)) {
			occluderLayers.push(layer.name);
		} else if (isIgnoredByDefault) {
			ignoredLayers.push(layer.name);
		} else if (isSamplingByDefault) {
			samplingLayers.push(layer.name);
		} else if (isOccluderByDefault) {
			occluderLayers.push(layer.name);
		} else {
			unknownLayers.push(layer.name);
		}
	}

	return { samplingLayers, occluderLayers, ignoredLayers, unknownLayers };
}

export function computeBoundsForLayers(
	layers: readonly LayerInventoryEntry[],
	layerNames: readonly string[],
	sampleHeight: number
): AnalysisBounds {
	const layerByName = new Map(layers.map((layer) => [layer.name, layer]));
	const missing = layerNames.filter((name) => !layerByName.has(name));
	if (missing.length > 0) {
		throw new Error(`Missing requested bounds layers: ${missing.join(', ')}`);
	}

	let bounds: LayerBounds | null = null;
	const emptyLayers: string[] = [];
	for (const name of layerNames) {
		const layer = layerByName.get(name);
		if (!layer?.bounds) {
			emptyLayers.push(name);
			continue;
		}
		bounds = mergeBounds(bounds, layer.bounds);
	}

	if (emptyLayers.length > 0) {
		throw new Error(`Requested bounds layers have no geometry bounds: ${emptyLayers.join(', ')}`);
	}
	if (!bounds) {
		throw new Error('No bounds layers selected');
	}

	return {
		x_min: bounds.minX,
		x_max: bounds.maxX,
		y_min: bounds.minY,
		y_max: bounds.maxY,
		z: sampleHeight
	};
}

export function computeCanonicalGridPointCount(params: {
	bounds: AnalysisBounds;
	gridSize: number;
	coordinateSystem: CoordinateSystem;
}): number {
	return canonicalGridPoints(params).numPoints;
}

function parseGlb(buffer: Buffer): ParsedGlb {
	if (buffer.toString('ascii', 0, 4) !== 'glTF') {
		throw new Error('Invalid GLB magic header');
	}
	const version = buffer.readUInt32LE(4);
	if (version !== 2) {
		throw new Error(`Unsupported GLB version: ${version}`);
	}

	let offset = 12;
	let json: GlbJson | null = null;
	let binChunk: Buffer | null = null;
	while (offset < buffer.length) {
		const chunkLength = buffer.readUInt32LE(offset);
		const chunkType = buffer.toString('ascii', offset + 4, offset + 8);
		const chunkStart = offset + 8;
		const chunkEnd = chunkStart + chunkLength;

		if (chunkType === 'JSON') {
			json = JSON.parse(buffer.toString('utf8', chunkStart, chunkEnd).trim());
		} else if (chunkType === 'BIN\0') {
			binChunk = buffer.subarray(chunkStart, chunkEnd);
		}

		offset = chunkEnd;
	}

	if (!json) {
		throw new Error('GLB is missing JSON chunk');
	}
	if (!binChunk) {
		throw new Error('GLB is missing BIN chunk');
	}
	return { json, binChunk };
}

function buildLayerInventory(parsed: ParsedGlb): GlbLayerInventory {
	const scene = parsed.json.scenes?.[parsed.json.scene ?? 0];
	const rootNodes = scene?.nodes ?? [];
	const layers: LayerInventoryEntry[] = [];

	for (const nodeIndex of rootNodes) {
		const node = getNode(parsed.json, nodeIndex);
		if (shouldSplitWrapperRootIntoChildLayers(parsed, nodeIndex)) {
			const rootMatrix = multiplyMatrices(identityMatrix(), nodeMatrix(node));
			for (const childIndex of node.children ?? []) {
				if (!subtreeHasPositionPrimitive(parsed, childIndex)) {
					continue;
				}
				const childNode = getNode(parsed.json, childIndex);
				const layer = createLayerEntry(childNode.name ?? `node_${childIndex}`);
				traverseNode(parsed, childIndex, rootMatrix, layer);
				layers.push(layer);
			}
		} else {
			const layer = createLayerEntry(node.name ?? `node_${nodeIndex}`);
			traverseNode(parsed, nodeIndex, identityMatrix(), layer);
			layers.push(layer);
		}
	}

	return { layers };
}

function shouldSplitWrapperRootIntoChildLayers(parsed: ParsedGlb, nodeIndex: number): boolean {
	const node = getNode(parsed.json, nodeIndex);
	if (node.mesh !== undefined) {
		return false;
	}

	const children = node.children ?? [];
	if (children.length < 2) {
		return false;
	}

	let namedGeometryChildCount = 0;
	for (const childIndex of children) {
		const childNode = getNode(parsed.json, childIndex);
		if (childNode.name && subtreeHasPositionPrimitive(parsed, childIndex)) {
			namedGeometryChildCount++;
		}
	}
	return namedGeometryChildCount >= 2;
}

function subtreeHasPositionPrimitive(parsed: ParsedGlb, nodeIndex: number): boolean {
	const node = getNode(parsed.json, nodeIndex);
	if (node.mesh !== undefined) {
		const mesh = getMesh(parsed.json, node.mesh);
		if ((mesh.primitives ?? []).some((primitive) => primitive.attributes?.POSITION !== undefined)) {
			return true;
		}
	}

	return (node.children ?? []).some((childIndex) => subtreeHasPositionPrimitive(parsed, childIndex));
}

function traverseNode(parsed: ParsedGlb, nodeIndex: number, parentMatrix: number[], layer: LayerInventoryEntry): void {
	const node = getNode(parsed.json, nodeIndex);
	const worldMatrix = multiplyMatrices(parentMatrix, nodeMatrix(node));

	if (node.mesh !== undefined) {
		const mesh = getMesh(parsed.json, node.mesh);
		for (const primitive of mesh.primitives ?? []) {
			addPrimitiveToLayer(parsed, primitive, worldMatrix, layer);
		}
	}

	for (const child of node.children ?? []) {
		traverseNode(parsed, child, worldMatrix, layer);
	}
}

function addPrimitiveToLayer(
	parsed: ParsedGlb,
	primitive: GlbPrimitive,
	worldMatrix: number[],
	layer: LayerInventoryEntry
): void {
	const positionAccessorIndex = primitive.attributes?.POSITION;
	if (positionAccessorIndex === undefined) {
		return;
	}

	layer.primitiveCount++;
	const mode = primitive.mode ?? 4;
	if (TRIANGLE_MODES.has(mode)) {
		layer.surfacePrimitiveCount++;
	} else if (LINE_MODES.has(mode)) {
		layer.linePrimitiveCount++;
	} else if (mode === POINT_MODE) {
		layer.pointPrimitiveCount++;
	}

	const primitiveBounds = computeAccessorBounds(parsed, positionAccessorIndex, worldMatrix);
	layer.bounds = mergeBounds(layer.bounds, primitiveBounds);
}

function computeAccessorBounds(parsed: ParsedGlb, accessorIndex: number, worldMatrix: number[]): LayerBounds {
	const accessor = getAccessor(parsed.json, accessorIndex);
	if (accessor.min && accessor.max && accessor.min.length >= 3 && accessor.max.length >= 3) {
		return transformAccessorMinMax(accessor.min, accessor.max, worldMatrix);
	}
	if (accessor.sparse) {
		throw new Error('Sparse POSITION accessors without min/max are not supported');
	}
	return computeAccessorBoundsFromBuffer(parsed, accessor, worldMatrix);
}

function computeAccessorBoundsFromBuffer(parsed: ParsedGlb, accessor: GlbAccessor, worldMatrix: number[]): LayerBounds {
	if (accessor.type !== 'VEC3') {
		throw new Error(`POSITION accessor must be VEC3, received ${accessor.type}`);
	}
	if (accessor.componentType !== 5126) {
		throw new Error(`POSITION accessor must use FLOAT components, received ${accessor.componentType}`);
	}
	if (accessor.bufferView === undefined) {
		throw new Error('POSITION accessor is missing bufferView');
	}

	const bufferView = getBufferView(parsed.json, accessor.bufferView);
	const componentBytes = componentByteSize(accessor.componentType);
	const componentCount = typeComponentCount(accessor.type);
	const stride = bufferView.byteStride ?? componentBytes * componentCount;
	const baseOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
	let bounds: LayerBounds | null = null;

	for (let index = 0; index < accessor.count; index++) {
		const offset = baseOffset + index * stride;
		const point = transformPoint(
			[
				parsed.binChunk.readFloatLE(offset),
				parsed.binChunk.readFloatLE(offset + componentBytes),
				parsed.binChunk.readFloatLE(offset + componentBytes * 2)
			],
			worldMatrix
		);
		bounds = mergePoint(bounds, point);
	}

	if (!bounds) {
		throw new Error('POSITION accessor has no vertices');
	}
	return bounds;
}

function transformAccessorMinMax(min: number[], max: number[], matrix: number[]): LayerBounds {
	let bounds: LayerBounds | null = null;
	for (const x of [min[0], max[0]]) {
		for (const y of [min[1], max[1]]) {
			for (const z of [min[2], max[2]]) {
				bounds = mergePoint(bounds, transformPoint([x, y, z], matrix));
			}
		}
	}
	if (!bounds) {
		throw new Error('Unable to transform accessor bounds');
	}
	return bounds;
}

function createLayerEntry(name: string): LayerInventoryEntry {
	return {
		name,
		bounds: null,
		primitiveCount: 0,
		surfacePrimitiveCount: 0,
		linePrimitiveCount: 0,
		pointPrimitiveCount: 0
	};
}

function normalizeLayerName(name: string): string {
	return name
		.trim()
		.toLowerCase()
		.replace(/[\s-]+/g, '_')
		.replace(/\(|\)/g, '');
}

function mergeBounds(current: LayerBounds | null, next: LayerBounds): LayerBounds {
	if (!current) {
		return { ...next };
	}
	return {
		minX: Math.min(current.minX, next.minX),
		maxX: Math.max(current.maxX, next.maxX),
		minY: Math.min(current.minY, next.minY),
		maxY: Math.max(current.maxY, next.maxY),
		minZ: Math.min(current.minZ, next.minZ),
		maxZ: Math.max(current.maxZ, next.maxZ)
	};
}

function mergePoint(current: LayerBounds | null, point: readonly [number, number, number]): LayerBounds {
	const [x, y, z] = point;
	if (!current) {
		return { minX: x, maxX: x, minY: y, maxY: y, minZ: z, maxZ: z };
	}
	return {
		minX: Math.min(current.minX, x),
		maxX: Math.max(current.maxX, x),
		minY: Math.min(current.minY, y),
		maxY: Math.max(current.maxY, y),
		minZ: Math.min(current.minZ, z),
		maxZ: Math.max(current.maxZ, z)
	};
}

function identityMatrix(): number[] {
	return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function nodeMatrix(node: GlbNode): number[] {
	if (node.matrix) {
		return [...node.matrix];
	}

	const translation = node.translation ?? [0, 0, 0];
	const rotation = node.rotation ?? [0, 0, 0, 1];
	const scale = node.scale ?? [1, 1, 1];
	return composeTrsMatrix(translation, rotation, scale);
}

function composeTrsMatrix(translation: number[], rotation: number[], scale: number[]): number[] {
	const [x, y, z, w] = rotation;
	const x2 = x + x;
	const y2 = y + y;
	const z2 = z + z;
	const xx = x * x2;
	const xy = x * y2;
	const xz = x * z2;
	const yy = y * y2;
	const yz = y * z2;
	const zz = z * z2;
	const wx = w * x2;
	const wy = w * y2;
	const wz = w * z2;
	const sx = scale[0];
	const sy = scale[1];
	const sz = scale[2];

	return [
		(1 - (yy + zz)) * sx,
		(xy + wz) * sx,
		(xz - wy) * sx,
		0,
		(xy - wz) * sy,
		(1 - (xx + zz)) * sy,
		(yz + wx) * sy,
		0,
		(xz + wy) * sz,
		(yz - wx) * sz,
		(1 - (xx + yy)) * sz,
		0,
		translation[0],
		translation[1],
		translation[2],
		1
	];
}

function multiplyMatrices(a: number[], b: number[]): number[] {
	const result = new Array<number>(16).fill(0);
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			result[col * 4 + row] =
				a[0 * 4 + row] * b[col * 4 + 0] +
				a[1 * 4 + row] * b[col * 4 + 1] +
				a[2 * 4 + row] * b[col * 4 + 2] +
				a[3 * 4 + row] * b[col * 4 + 3];
		}
	}
	return result;
}

function transformPoint(point: readonly [number, number, number], matrix: number[]): [number, number, number] {
	const [x, y, z] = point;
	return [
		matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
		matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
		matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
	];
}

function getNode(json: GlbJson, index: number): GlbNode {
	const node = json.nodes?.[index];
	if (!node) {
		throw new Error(`Missing node ${index}`);
	}
	return node;
}

function getMesh(json: GlbJson, index: number): GlbMesh {
	const mesh = json.meshes?.[index];
	if (!mesh) {
		throw new Error(`Missing mesh ${index}`);
	}
	return mesh;
}

function getAccessor(json: GlbJson, index: number): GlbAccessor {
	const accessor = json.accessors?.[index];
	if (!accessor) {
		throw new Error(`Missing accessor ${index}`);
	}
	return accessor;
}

function getBufferView(json: GlbJson, index: number): GlbBufferView {
	const bufferView = json.bufferViews?.[index];
	if (!bufferView) {
		throw new Error(`Missing bufferView ${index}`);
	}
	if (bufferView.buffer !== 0) {
		throw new Error(`Only single-BIN GLB buffers are supported; received buffer ${bufferView.buffer}`);
	}
	return bufferView;
}

function componentByteSize(componentType: number): number {
	const size = COMPONENT_BYTE_SIZE[componentType];
	if (!size) {
		throw new Error(`Unsupported accessor component type: ${componentType}`);
	}
	return size;
}

function typeComponentCount(type: string): number {
	const count = TYPE_COMPONENTS[type];
	if (!count) {
		throw new Error(`Unsupported accessor type: ${type}`);
	}
	return count;
}

async function generateMetadata(options: CliOptions): Promise<void> {
	const inventory = await readGlbLayerInventory(options.model);
	const roles = inferLayerRoles(inventory.layers, {
		samplingLayers: options.samplingLayers,
		occluderLayers: options.occluderLayers,
		ignoredLayers: options.ignoredLayers
	});
	const bounds = computeBoundsForLayers(inventory.layers, roles.samplingLayers, options.sampleHeight);
	const numPositions = computeCanonicalGridPointCount({
		bounds,
		gridSize: options.gridSize,
		coordinateSystem: options.coordinateSystem
	});
	const weather = resolveWeatherProfile(options.weatherProfile);
	const metadata = {
		analysis_id: options.analysisId,
		date: options.date,
		grid_size: options.gridSize,
		analysis_type: 'full_day',
		hours: Array.from({ length: 24 }, (_, hour) => hour),
		bounds,
		utci_range: { min: 0, max: 0, mean: 0, std: 0 },
		num_positions: numPositions,
		model_file: toDataModelPath(options.projectId, options.model),
		epw_file: weather.epw_file,
		location: weather.location,
		generation_date: new Date().toISOString(),
		runtime_seconds: 0,
		coordinate_system: options.coordinateSystem,
		hour_statistics: [],
		has_shading_index: false,
		shading_index_range: { min: 0, max: 1 }
	};

	await mkdir(path.dirname(options.out), { recursive: true });
	await writeFile(options.out, `${JSON.stringify(metadata, null, 2)}\n`);
	printValidationReport(roles, bounds, numPositions);
}

function resolveWeatherProfile(profile: string): typeof BEER_SHEVA_WEATHER {
	if (profile !== 'beer-sheva') {
		throw new Error(`Unsupported weather profile: ${profile}`);
	}
	return BEER_SHEVA_WEATHER;
}

function toDataModelPath(projectId: string, modelPath: string): string {
	return path.posix.join('data/3d_models', projectId, path.basename(modelPath));
}

function printValidationReport(roles: LayerRoles, bounds: AnalysisBounds, numPositions: number): void {
	console.log('Live WebGPU metadata validation report');
	console.log(`  sampling layers: ${formatList(roles.samplingLayers)}`);
	console.log(`  occluder layers: ${formatList(roles.occluderLayers)}`);
	console.log(`  ignored layers: ${formatList(roles.ignoredLayers)}`);
	console.log(`  unknown layers: ${formatList(roles.unknownLayers)}`);
	console.log(`  bounds: ${JSON.stringify(bounds)}`);
	console.log(`  point count: ${numPositions}`);
}

function formatList(values: readonly string[]): string {
	return values.length > 0 ? values.join(', ') : '(none)';
}

function parseCliArgs(argv: string[]): CliOptions {
	const values = new Map<string, string>();
	for (let index = 0; index < argv.length; index++) {
		const key = argv[index];
		if (!key?.startsWith('--')) {
			throw new Error(`Unexpected argument: ${key}`);
		}
		const value = argv[index + 1];
		if (!value || value.startsWith('--')) {
			throw new Error(`Missing value for ${key}`);
		}
		values.set(key.slice(2), value);
		index++;
	}

	return {
		model: requireOption(values, 'model'),
		out: requireOption(values, 'out'),
		analysisId: requireOption(values, 'analysis-id'),
		projectId: requireOption(values, 'project-id'),
		gridSize: parsePositiveFiniteNumberOption(values, 'grid-size'),
		date: requireOption(values, 'date'),
		coordinateSystem: parseCoordinateSystem(requireOption(values, 'coordinate-system')),
		sampleHeight: parseFiniteNumberOption(values, 'sample-height'),
		weatherProfile: requireOption(values, 'weather-profile'),
		samplingLayers: parseLayerList(values.get('sampling-layers')),
		occluderLayers: parseLayerList(values.get('occluder-layers')),
		ignoredLayers: parseLayerList(values.get('ignored-layers'))
	};
}

function requireOption(values: Map<string, string>, key: string): string {
	const value = values.get(key);
	if (!value) {
		throw new Error(`Missing required option --${key}`);
	}
	return value;
}

function parseCoordinateSystem(value: string): CoordinateSystem {
	if (value === 'xy_ground' || value === 'xz_ground') {
		return value;
	}
	throw new Error(`Unsupported coordinate system: ${value}`);
}

function parsePositiveFiniteNumberOption(values: Map<string, string>, key: string): number {
	const rawValue = requireOption(values, key);
	const value = Number(rawValue);
	if (rawValue.trim() === '' || !Number.isFinite(value) || value <= 0) {
		throw new Error(`--${key} must be a finite positive number, received "${rawValue}"`);
	}
	return value;
}

function parseFiniteNumberOption(values: Map<string, string>, key: string): number {
	const rawValue = requireOption(values, key);
	const value = Number(rawValue);
	if (rawValue.trim() === '' || !Number.isFinite(value)) {
		throw new Error(`--${key} must be a finite number, received "${rawValue}"`);
	}
	return value;
}

function parseLayerList(value: string | undefined): string[] | undefined {
	if (!value) {
		return undefined;
	}
	return value
		.split(',')
		.map((item) => item.trim())
		.filter(Boolean);
}

async function main(): Promise<void> {
	await generateMetadata(parseCliArgs(process.argv.slice(2)));
}

const isCli = process.argv[1]
	? import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
	: false;

if (isCli) {
	main().catch((error: unknown) => {
		console.error(error instanceof Error ? error.message : error);
		process.exitCode = 1;
	});
}
