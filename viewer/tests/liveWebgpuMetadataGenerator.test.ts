import { execFile } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { describe, expect, it, beforeAll } from 'vitest';
import {
	computeBoundsForLayers,
	computeCanonicalGridPointCount,
	inferLayerRoles,
	readGlbLayerInventory
} from '../scripts/generate-live-webgpu-metadata';

const MODEL_PATH = '../data/3d_models/Innovation-District/innovation_district.glb';
const execFileAsync = promisify(execFile);

describe('live WebGPU metadata generator', () => {
	let inventory: Awaited<ReturnType<typeof readGlbLayerInventory>>;

	beforeAll(async () => {
		inventory = await readGlbLayerInventory(MODEL_PATH);
	});

	it('returns the expected Innovation District raw top-level layers', () => {
		const layerNames = inventory.layers.map((layer) => layer.name);

		expect(layerNames).toEqual(expect.arrayContaining([
			'trees_canopy',
			'existing_buildings',
			'ground',
			'district_outline',
			'street',
			'train_tracks'
		]));
		expect(layerNames).not.toContain('trees_point');
	});

	it('infers Innovation District sampling, occluder, ignored, and unknown layers', () => {
		const roles = inferLayerRoles(inventory.layers);

		expect(roles.samplingLayers).toEqual(['ground', 'street', 'train_tracks']);
		expect(roles.occluderLayers).toEqual(['trees_canopy', 'existing_buildings']);
		expect(roles.ignoredLayers).toEqual(['district_outline']);
		expect(roles.unknownLayers).toEqual([]);
	});

	it('ignores future tree point layers instead of treating them as occluders', () => {
		const roles = inferLayerRoles([
			{
				name: 'trees_point',
				bounds: null,
				primitiveCount: 1,
				surfacePrimitiveCount: 1,
				linePrimitiveCount: 0,
				pointPrimitiveCount: 0
			},
			{
				name: 'tree_point',
				bounds: null,
				primitiveCount: 1,
				surfacePrimitiveCount: 1,
				linePrimitiveCount: 0,
				pointPrimitiveCount: 0
			}
		]);

		expect(roles.samplingLayers).toEqual([]);
		expect(roles.occluderLayers).toEqual([]);
		expect(roles.ignoredLayers).toEqual(['trees_point', 'tree_point']);
		expect(roles.unknownLayers).toEqual([]);
	});

	it('applies partial role overrides without disabling default inference', () => {
		const roles = inferLayerRoles(inventory.layers, {
			ignoredLayers: ['street'],
			occluderLayers: ['train_tracks']
		});

		expect(roles.samplingLayers).toEqual(['ground']);
		expect(roles.occluderLayers).toEqual(['trees_canopy', 'existing_buildings', 'train_tracks']);
		expect(roles.ignoredLayers).toEqual(['district_outline', 'street']);
		expect(roles.unknownLayers).toEqual([]);
	});

	it('surfaces unknown future non-surface layers instead of auto-ignoring them', () => {
		const roles = inferLayerRoles([
			{
				name: 'future_lines',
				bounds: null,
				primitiveCount: 1,
				surfacePrimitiveCount: 0,
				linePrimitiveCount: 1,
				pointPrimitiveCount: 0
			},
			{
				name: 'future_points',
				bounds: null,
				primitiveCount: 1,
				surfacePrimitiveCount: 0,
				linePrimitiveCount: 0,
				pointPrimitiveCount: 1
			}
		]);

		expect(roles.samplingLayers).toEqual([]);
		expect(roles.occluderLayers).toEqual([]);
		expect(roles.ignoredLayers).toEqual([]);
		expect(roles.unknownLayers).toEqual(['future_lines', 'future_points']);
	});

	it('computes sampling bounds from ground, street, and train_tracks', () => {
		const bounds = computeBoundsForLayers(inventory.layers, ['ground', 'street', 'train_tracks'], 1.5);

		expect(bounds.x_min).toBeCloseTo(180591.05, 2);
		expect(bounds.x_max).toBeCloseTo(183188.48, 2);
		expect(bounds.y_min).toBeCloseTo(573608.4, 2);
		expect(bounds.y_max).toBeCloseTo(575905.44, 2);
		expect(bounds.z).toBe(1.5);
	});

	it('derives the canonical grid point count for grid size 2', () => {
		const bounds = computeBoundsForLayers(inventory.layers, ['ground', 'street', 'train_tracks'], 1.5);

		expect(
			computeCanonicalGridPointCount({
				bounds,
				gridSize: 2,
				coordinateSystem: 'xy_ground'
			})
		).toBe(1_492_551);
	});

	it('throws a clear error when requested bounds layers are missing', () => {
		expect(() => computeBoundsForLayers(inventory.layers, ['missing_layer'], 1.5)).toThrow(
			/Missing requested bounds layers: missing_layer/
		);
	});

	it('rejects sparse POSITION accessors without accessor min/max', async () => {
		const dir = await mkdtemp(path.join(tmpdir(), 'live-webgpu-glb-'));
		const glbPath = path.join(dir, 'sparse.glb');
		await writeFile(glbPath, createSparsePositionGlbWithoutMinMax());

		try {
			await expect(readGlbLayerInventory(glbPath)).rejects.toThrow(
				/Sparse POSITION accessors without min\/max are not supported/
			);
		} finally {
			await rm(dir, { recursive: true, force: true });
		}
	});

	it('discovers named semantic child layers below a wrapper root', async () => {
		const dir = await mkdtemp(path.join(tmpdir(), 'live-webgpu-glb-'));
		const glbPath = path.join(dir, 'wrapped.glb');
		await writeFile(glbPath, createWrapperRootGlbWithChildLayers());

		try {
			const wrappedInventory = await readGlbLayerInventory(glbPath);

			expect(wrappedInventory.layers.map((layer) => layer.name)).toEqual([
				'ground',
				'existing_buildings'
			]);
			expect(wrappedInventory.layers.find((layer) => layer.name === 'ground')?.surfacePrimitiveCount).toBe(1);
			expect(
				wrappedInventory.layers.find((layer) => layer.name === 'existing_buildings')?.surfacePrimitiveCount
			).toBe(1);
		} finally {
			await rm(dir, { recursive: true, force: true });
		}
	});

	it('rejects non-finite grid size values in the CLI', async () => {
		const failure = await runGeneratorCliExpectingFailure(['--grid-size', 'NaN']);

		expect(failure.stderr).toContain('--grid-size must be a finite positive number');
	});

	it('rejects non-finite sample height values in the CLI', async () => {
		const failure = await runGeneratorCliExpectingFailure(['--sample-height', 'Infinity']);

		expect(failure.stderr).toContain('--sample-height must be a finite number');
	});
});

async function runGeneratorCliExpectingFailure(overrides: string[]): Promise<{ stdout: string; stderr: string }> {
	const args = [
		path.resolve(__dirname, '../node_modules/tsx/dist/cli.mjs'),
		'scripts/generate-live-webgpu-metadata.ts',
		'--model',
		MODEL_PATH,
		'--out',
		path.join(tmpdir(), 'live-webgpu-metadata-invalid.json'),
		'--analysis-id',
		'innovation_district_webgpu',
		'--project-id',
		'Innovation-District',
		'--grid-size',
		'2',
		'--date',
		'20250815',
		'--coordinate-system',
		'xy_ground',
		'--sample-height',
		'1.5',
		'--weather-profile',
		'beer-sheva'
	];

	for (let index = 0; index < overrides.length; index += 2) {
		const optionIndex = args.indexOf(overrides[index]);
		if (optionIndex === -1) {
			throw new Error(`Unknown test override option: ${overrides[index]}`);
		}
		args[optionIndex + 1] = overrides[index + 1];
	}

	try {
		await execFileAsync(process.execPath, args, {
			cwd: path.resolve(__dirname, '..')
		});
		throw new Error('Expected generator CLI to fail');
	} catch (error) {
		if (error && typeof error === 'object' && 'stderr' in error && 'stdout' in error) {
			return {
				stdout: String(error.stdout),
				stderr: String(error.stderr)
			};
		}
		throw error;
	}
}

function createWrapperRootGlbWithChildLayers(): Buffer {
	const json = {
		asset: { version: '2.0' },
		scene: 0,
		scenes: [{ nodes: [0] }],
		nodes: [
			{ name: 'model_wrapper', children: [1, 2] },
			{ name: 'ground', mesh: 0 },
			{ name: 'existing_buildings', mesh: 1 }
		],
		meshes: [
			{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] },
			{ primitives: [{ attributes: { POSITION: 1 }, mode: 4 }] }
		],
		accessors: [
			{ componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 0, 1] },
			{ componentType: 5126, count: 3, type: 'VEC3', min: [10, 0, 10], max: [11, 4, 11] }
		],
		buffers: [{ byteLength: 4 }]
	};
	const jsonChunk = paddedChunk(Buffer.from(JSON.stringify(json), 'utf8'), 'JSON', 0x20);
	const binChunk = paddedChunk(Buffer.alloc(4), 'BIN\0', 0);
	const header = Buffer.alloc(12);
	header.write('glTF', 0, 'ascii');
	header.writeUInt32LE(2, 4);
	header.writeUInt32LE(header.length + jsonChunk.length + binChunk.length, 8);
	return Buffer.concat([header, jsonChunk, binChunk]);
}

function createSparsePositionGlbWithoutMinMax(): Buffer {
	const json = {
		asset: { version: '2.0' },
		scene: 0,
		scenes: [{ nodes: [0] }],
		nodes: [{ name: 'sparse_surface', mesh: 0 }],
		meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] }],
		accessors: [
			{
				bufferView: 0,
				componentType: 5126,
				count: 1,
				type: 'VEC3',
				sparse: {
					count: 1,
					indices: { bufferView: 1, componentType: 5123 },
					values: { bufferView: 2 }
				}
			}
		],
		bufferViews: [
			{ buffer: 0, byteOffset: 0, byteLength: 12 },
			{ buffer: 0, byteOffset: 12, byteLength: 2 },
			{ buffer: 0, byteOffset: 16, byteLength: 12 }
		],
		buffers: [{ byteLength: 28 }]
	};
	const jsonChunk = paddedChunk(Buffer.from(JSON.stringify(json), 'utf8'), 'JSON', 0x20);
	const binChunk = paddedChunk(Buffer.alloc(28), 'BIN\0', 0);
	const header = Buffer.alloc(12);
	header.write('glTF', 0, 'ascii');
	header.writeUInt32LE(2, 4);
	header.writeUInt32LE(header.length + jsonChunk.length + binChunk.length, 8);
	return Buffer.concat([header, jsonChunk, binChunk]);
}

function paddedChunk(payload: Buffer, type: string, padByte: number): Buffer {
	const paddingLength = (4 - (payload.length % 4)) % 4;
	const chunk = Buffer.alloc(8 + payload.length + paddingLength, padByte);
	chunk.writeUInt32LE(payload.length + paddingLength, 0);
	chunk.write(type, 4, 'ascii');
	payload.copy(chunk, 8);
	return chunk;
}
