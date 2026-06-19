import { describe, expect, it } from 'vitest';
import { join, resolve } from 'node:path';

import {
	type ActiveCellArrayBuildParams,
	type RawExportMetadataBuildParams,
	buildActiveCellArrays,
	buildRawExportMetadata
} from '$lib/gis/innovationDistrictExport';
import {
	buildPublicationTimeoutErrorMessage,
	buildRawExportFileNames,
	parseExportArgs
} from '../../scripts/export-innovation-district-gis';
import {
	resolveRepoRelativePath,
	resolveRepoRoot
} from '../../scripts/repo-paths';

const RAW_SOURCE_FIELDS = {
	sourceAnalysisId: 'Innovation-District/innovation_district_webgpu',
	sourceModelPath: 'data/3d_models/Innovation-District/innovation_district.glb',
	sourceGeorefPath: 'data/3d_models/Innovation-District/innovation_district.georef.json',
	declaredCrs: 'EPSG:2039',
	gridSize: 2
};

const repoRoot = resolve('tmp', 'fast-utci-repo');
const viewerRoot = join(repoRoot, 'viewer');
const familyOutDirArg = join('..', 'data', 'gis', 'Innovation-District');
const bundleOutDirArg = join('..', 'data', 'gis', 'Innovation-District', '2025-08-15_2m');
const dataOutDirArg = join('data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw');
const parentRelativeDataOutDirArg = join('..', 'data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw');
const relativeOutDirArg = join('tmp-gis-smoke');
const canonicalFamilyOutDir = join(repoRoot, 'data', 'gis', 'Innovation-District');
const canonicalBundleOutDir = join(canonicalFamilyOutDir, '2025-08-15_2m');
const canonicalRawOutDir = join(canonicalBundleOutDir, 'raw');
const customAbsoluteFamilySuffixOutDir = resolve(
	repoRoot,
	'..',
	'custom-exports',
	'data',
	'gis',
	'Innovation-District'
);

function buildRuntimeInvalidActiveCellParams(
	overrides: Partial<Record<string, unknown>>
): ActiveCellArrayBuildParams {
	return {
		activeCanonicalIndices: new Uint32Array([5, 2, 9]),
		positions: new Float32Array([
			180605, 573605, 1.5,
			180602, 573602, 1.5,
			180609, 573609, 1.5
		]),
		utciByHour: [new Float32Array([35, 32, 40])],
		shadingIndex: new Float32Array([0.5, 0.25, 0.75]),
		surfaceFlags: new Uint8Array([1, 3, 6]),
		hours: [0],
		activeMaskSource: 'base+road',
		...(overrides as Partial<ActiveCellArrayBuildParams>)
	};
}

function buildRuntimeInvalidRawMetadataParams(
	overrides: Partial<Record<string, unknown>>
): RawExportMetadataBuildParams {
	return {
		schemaVersion: 'innovation-district-raw-export/v1',
		analysisId: 'Innovation-District/2025-08-15_2m_fullday',
		...RAW_SOURCE_FIELDS,
		coordinateSystem: 'projected-analysis',
		hours: [0],
		canonicalIndices: new Uint32Array([5, 2, 9]),
		positions: new Float32Array([
			180605, 573605, 1.5,
			180602, 573602, 1.5,
			180609, 573609, 1.5
		]),
		utci: new Float32Array([35, 32, 40]),
		shadingIndex: new Float32Array([0.5, 0.25, 0.75]),
		surfaceFlags: new Uint8Array([1, 3, 6]),
		activeMask: {
			source: 'base+road',
			canonicalPointCount: 12,
			checksum: 'mask-sha256',
			signature: 'mask-signature'
		},
		...(overrides as Partial<RawExportMetadataBuildParams>)
	};
}

// @ts-expect-error surfaceFlags is required on trusted classified export builder params.
const _missingSurfaceFlagsActiveBuilderParams: ActiveCellArrayBuildParams = {
	activeCanonicalIndices: new Uint32Array([5, 2, 9]),
	positions: new Float32Array([
		180605, 573605, 1.5,
		180602, 573602, 1.5,
		180609, 573609, 1.5
	]),
	utciByHour: [new Float32Array([35, 32, 40])],
	shadingIndex: new Float32Array([0.5, 0.25, 0.75]),
	hours: [0],
	activeMaskSource: 'base+road'
};

// @ts-expect-error surfaceFlags is required on trusted raw metadata builder params.
const _missingSurfaceFlagsRawMetadataParams: RawExportMetadataBuildParams = {
	schemaVersion: 'innovation-district-raw-export/v1',
	analysisId: 'Innovation-District/2025-08-15_2m_fullday',
	...RAW_SOURCE_FIELDS,
	coordinateSystem: 'projected-analysis',
	hours: [0],
	canonicalIndices: new Uint32Array([5, 2, 9]),
	positions: new Float32Array([
		180605, 573605, 1.5,
		180602, 573602, 1.5,
		180609, 573609, 1.5
	]),
	utci: new Float32Array([35, 32, 40]),
	shadingIndex: new Float32Array([0.5, 0.25, 0.75]),
	activeMask: {
		source: 'base+road',
		canonicalPointCount: 12,
		checksum: 'mask-sha256',
		signature: 'mask-signature'
	}
};

describe('parseExportArgs', () => {
	it('recovers npm package-script flags when npm 11 strips option names into config env', () => {
		const args = parseExportArgs({
			argv: [relativeOutDirArg, '0'],
			cwd: viewerRoot,
			env: {
				npm_lifecycle_event: 'gis:export-innovation-district',
				npm_config_dry_run_args: 'true',
				npm_config_out_dir: 'true',
				npm_config_hour: 'true',
				npm_config_headless: 'true'
			}
		});

		expect(args.outDir).toBe(resolve(viewerRoot, relativeOutDirArg));
		expect(args.hours).toEqual([0]);
		expect(args.headless).toBe(true);
		expect(args.dryRunArgs).toBe(true);
	});

	it('resolves data-prefixed out-dir paths against the repo root when run from viewer', () => {
		const args = parseExportArgs({
			argv: ['--out-dir', dataOutDirArg],
			cwd: viewerRoot,
			env: {}
		});

		expect(args.outDir).toBe(
			join(repoRoot, 'data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw')
		);
	});

	it('resolves data-prefixed out-dir paths against the repo root when run from repo root', () => {
		const args = parseExportArgs({
			argv: ['--out-dir', dataOutDirArg],
			cwd: repoRoot,
			env: {}
		});

		expect(args.outDir).toBe(
			join(repoRoot, 'data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw')
		);
	});

	it('keeps absolute out-dir paths absolute', () => {
		const absoluteOutDir = resolve(repoRoot, '..', 'exports', 'Innovation-District', 'raw');
		const args = parseExportArgs({
			argv: ['--out-dir', absoluteOutDir],
			cwd: viewerRoot,
			env: {}
		});

		expect(args.outDir).toBe(absoluteOutDir);
	});

	it('still accepts parent-relative data paths that land on the same repo-root export directory', () => {
		const args = parseExportArgs({
			argv: ['--out-dir', parentRelativeDataOutDirArg],
			cwd: viewerRoot,
			env: {}
		});

		expect(args.outDir).toBe(
			join(repoRoot, 'data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw')
		);
	});

	it('expands the canonical Innovation District bundle out-dir to its raw directory', () => {
		const args = parseExportArgs({
			argv: ['--out-dir', bundleOutDirArg],
			cwd: viewerRoot,
			env: {}
		});

		expect(args.outDir).toBe(canonicalRawOutDir);
	});

	it('expands the documented Innovation District family out-dir to the dated raw bundle directory', () => {
		const args = parseExportArgs({
			argv: ['--out-dir', familyOutDirArg],
			cwd: viewerRoot,
			env: {}
		});

		expect(args.outDir).toBe(canonicalRawOutDir);
	});

	it('does not rewrite a custom absolute path that only ends with the Innovation District family suffix', () => {
		const args = parseExportArgs({
			argv: ['--out-dir', customAbsoluteFamilySuffixOutDir],
			cwd: viewerRoot,
			env: {}
		});

		expect(args.outDir).toBe(customAbsoluteFamilySuffixOutDir);
	});

	it('keeps repo-root georef lookup stable from both viewer and repo-root cwd policies', () => {
		const expected = resolve(
			repoRoot,
			'data',
			'3d_models',
			'Innovation-District',
			'innovation_district.georef.json'
		);

		expect(resolveRepoRelativePath(viewerRoot, RAW_SOURCE_FIELDS.sourceGeorefPath)).toBe(expected);
		expect(resolveRepoRelativePath(repoRoot, RAW_SOURCE_FIELDS.sourceGeorefPath)).toBe(expected);
	});
});

describe('repo-path helpers', () => {
	it('resolves the repo root from viewer cwd and preserves repo-root cwd', () => {
		expect(resolveRepoRoot(viewerRoot)).toBe(repoRoot);
		expect(resolveRepoRoot(repoRoot)).toBe(repoRoot);
	});

	it('keeps non-repo-relative paths rooted at the provided cwd', () => {
		expect(resolveRepoRelativePath(viewerRoot, relativeOutDirArg)).toBe(
			resolve(viewerRoot, relativeOutDirArg)
		);
	});
});

describe('buildRawExportFileNames', () => {
	it('builds the planned date and grid-size prefixed raw artifact names', () => {
		expect(buildRawExportFileNames({ date: '2025-08-15', gridSize: 2 })).toEqual({
			metadata: '2025-08-15_2m_active-cells.metadata.json',
			canonicalIndices: '2025-08-15_2m_active-cells.canonical.u32.bin',
			positions: '2025-08-15_2m_active-cells.positions.f32.bin',
			utci: '2025-08-15_2m_active-cells.utci.f32.bin',
			shadingIndex: '2025-08-15_2m_active-cells.shading.f32.bin',
			surfaceFlags: '2025-08-15_2m_active-cells.surface-flags.u8.bin'
		});
	});
});

describe('buildPublicationTimeoutErrorMessage', () => {
	it('adds a clear actionable hint when headless Chromium never reaches the live WebGPU route state', () => {
		const message = buildPublicationTimeoutErrorMessage({
			metricType: 'utci',
			headless: true,
			error: new Error('page.waitForFunction: Timeout 180000ms exceeded.'),
			diagnostics: {
				utciRenderResolved: 'dataTexture',
				rendererBackend: 'unknown',
				baseRenderTransport: 'idle',
				baseLiveReady: false,
				selectedHourRuntimeContract: {
					renderTransport: 'none',
					utciSurfaceSource: 'none'
				}
			}
		});

		expect(message).toContain('Headless Chromium did not expose the required live WebGPU publication contract');
		expect(message).toContain('Rerun without --headless');
		expect(message).toContain('gpuNative / compute-buffer-selected-hour');
	});

	it('keeps non-headless publication timeouts generic', () => {
		const message = buildPublicationTimeoutErrorMessage({
			metricType: 'shading_index',
			headless: false,
			error: new Error('page.waitForFunction: Timeout 180000ms exceeded.'),
			diagnostics: {
				utciRenderResolved: 'dataTexture',
				rendererBackend: 'webgl',
				baseRenderTransport: 'cpu-uploaded-selected-hour',
				baseLiveReady: true
			}
		});

		expect(message).toContain('Timed out waiting for shading_index publication.');
		expect(message).not.toContain('Rerun without --headless');
	});
});

describe('buildActiveCellArrays', () => {
	it('exports only active compact rows and preserves canonical identity', () => {
		const result = buildActiveCellArrays({
			activeCanonicalIndices: new Uint32Array([2, 5]),
			positions: new Float32Array([
				180600, 573600, 1.5,
				180602, 573602, 1.5
			]),
			utciByHour: [new Float32Array([30, 32]), new Float32Array([31, 33])],
			shadingIndex: new Float32Array([0.25, 0.75]),
			surfaceFlags: new Uint8Array([1, 6]),
			hours: [0, 1],
			activeMaskSource: 'base+road'
		});

		expect(Array.from(result.canonicalIndices)).toEqual([2, 5]);
		expect(Array.from(result.positions)).toEqual([
			180600, 573600, 1.5, 180602, 573602, 1.5
		]);
		expect(Array.from(result.utci)).toEqual([30, 31, 32, 33]);
		expect(Array.from(result.shadingIndex)).toEqual([0.25, 0.75]);
		expect(Array.from(result.surfaceFlags)).toEqual([1, 6]);
		expect(result.layout.utci).toBe('point-major-hour');
		expect(result).not.toHaveProperty('metadata');
	});

	it('throws when a UTCI hour slice length does not match the active row count', () => {
		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2, 5]),
				positions: new Float32Array([
					180600, 573600, 1.5,
					180602, 573602, 1.5
				]),
				utciByHour: [new Float32Array([30, 32]), new Float32Array([31])],
				shadingIndex: new Float32Array([0.25, 0.75]),
				surfaceFlags: new Uint8Array([1, 6]),
				hours: [0, 1],
				activeMaskSource: 'base+road'
			})
		).toThrowError(/utciByHour\[1\]/i);
	});

	it('throws when shadingIndex length does not match the active row count', () => {
		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2, 5]),
				positions: new Float32Array([
					180600, 573600, 1.5,
					180602, 573602, 1.5
				]),
				utciByHour: [new Float32Array([30, 32]), new Float32Array([31, 33])],
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1, 6]),
				hours: [0, 1],
				activeMaskSource: 'base+road'
			})
		).toThrowError(/shadingIndex/i);
	});

	it('throws when activeCanonicalIndices are missing', () => {
		expect(() =>
			buildActiveCellArrays({
				positions: new Float32Array([
					180600, 573600, 1.5,
					180602, 573602, 1.5
				]),
				utciByHour: [new Float32Array([30, 32])],
				shadingIndex: new Float32Array([0.25, 0.75]),
				surfaceFlags: new Uint8Array([1, 6]),
				hours: [0],
				activeMaskSource: 'base+road'
			})
		).toThrowError(/activeCanonicalIndices/i);
	});

	it('throws when positions or values contain non-finite numbers', () => {
		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, Number.NaN, 1.5]),
				utciByHour: [new Float32Array([30])],
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				hours: [0],
				activeMaskSource: 'base+road'
			})
		).toThrowError(/positions/i);

		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utciByHour: [new Float32Array([Number.POSITIVE_INFINITY])],
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				hours: [0],
				activeMaskSource: 'base+road'
			})
		).toThrowError(/utci/i);

		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utciByHour: [new Float32Array([30])],
				shadingIndex: new Float32Array([Number.NaN]),
				surfaceFlags: new Uint8Array([1]),
				hours: [0],
				activeMaskSource: 'base+road'
			})
		).toThrowError(/shadingIndex/i);
	});

	it('throws when activeMaskSource is missing or unexpected', () => {
		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utciByHour: [new Float32Array([30])],
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				hours: [0]
			})
		).toThrowError(/activeMaskSource/i);

		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utciByHour: [new Float32Array([30])],
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				hours: [0],
				activeMaskSource: 'base'
			})
		).toThrowError(/base\+road/i);
	});

	it('requires classified surfaceFlags for the Innovation District classified export path and preserves active row order', () => {
		const params = buildRuntimeInvalidActiveCellParams({ surfaceFlags: undefined });

		expect(() => buildActiveCellArrays(params)).toThrowError(/surfaceFlags/i);
	});

	it('validates surfaceFlags length equals active row count and returns the aligned raw array plus layout metadata', () => {
		const paramsWithWrongSurfaceFlagLength = {
			activeCanonicalIndices: new Uint32Array([5, 2, 9]),
			positions: new Float32Array([
				180605, 573605, 1.5,
				180602, 573602, 1.5,
				180609, 573609, 1.5
			]),
			utciByHour: [new Float32Array([35, 32, 40])],
			shadingIndex: new Float32Array([0.5, 0.25, 0.75]),
			surfaceFlags: new Uint8Array([1, 3]),
			hours: [0],
			activeMaskSource: 'base+road' as const
		};

		expect(() => buildActiveCellArrays(paramsWithWrongSurfaceFlagLength)).toThrowError(
			/surfaceFlags/i
		);

		const params = {
			...paramsWithWrongSurfaceFlagLength,
			surfaceFlags: new Uint8Array([1, 3, 6])
		};
		const result = buildActiveCellArrays(params);

		expect(Array.from(result.canonicalIndices)).toEqual([5, 2, 9]);
		expect(Array.from(result.surfaceFlags)).toEqual([1, 3, 6]);
		expect(result.layout).toHaveProperty('surfaceFlags', 'point-major');
	});

	it('rejects classified active rows whose surfaceFlags omit both sampled-surface bits', () => {
		const params: Parameters<typeof buildActiveCellArrays>[0] = {
			activeCanonicalIndices: new Uint32Array([5, 2, 9]),
			positions: new Float32Array([
				180605, 573605, 1.5,
				180602, 573602, 1.5,
				180609, 573609, 1.5
			]),
			utciByHour: [new Float32Array([35, 32, 40])],
			shadingIndex: new Float32Array([0.5, 0.25, 0.75]),
			surfaceFlags: new Uint8Array([1, 0, 6]),
			hours: [0],
			activeMaskSource: 'base+road' as const
		};

		expect(() => buildActiveCellArrays(params)).toThrowError(/sampled-surface|surfaceFlags|unknown/i);
	});
});

describe('buildRawExportMetadata', () => {
	it('builds validated raw metadata including checksums, shapes, and optional timings', () => {
		const canonicalIndices = new Uint32Array([2, 5]);
		const positions = new Float32Array([
			180600, 573600, 1.5,
			180602, 573602, 1.5
		]);
		const utci = new Float32Array([30, 31, 32, 33]);
		const shadingIndex = new Float32Array([0.25, 0.75]);
		const surfaceFlags = new Uint8Array([1, 6]);

		const metadata = buildRawExportMetadata({
			schemaVersion: 'innovation-district-raw-export/v1',
			analysisId: 'Innovation-District/2025-08-15_2m_fullday',
			...RAW_SOURCE_FIELDS,
			coordinateSystem: 'projected-analysis',
			hours: [0, 1],
			canonicalIndices,
			positions,
			utci,
			shadingIndex,
			surfaceFlags,
			activeMask: {
				source: 'base+road',
				canonicalPointCount: 4,
				checksum: 'mask-sha256',
				signature: 'mask-signature'
			},
			files: {
				canonicalIndices: {
					fileName: '2025-08-15_2m_active-cells.canonical.u32.bin',
					checksum: 'canonical-sha256'
				},
				positions: {
					fileName: '2025-08-15_2m_active-cells.positions.f32.bin',
					checksum: 'positions-sha256'
				},
				utci: {
					fileName: '2025-08-15_2m_active-cells.utci.f32.bin',
					checksum: 'utci-sha256'
				},
				shadingIndex: {
					fileName: '2025-08-15_2m_active-cells.shading.f32.bin',
					checksum: 'shading-sha256'
				},
				surfaceFlags: {
					fileName: '2025-08-15_2m_active-cells.surface-flags.u8.bin',
					checksum: 'surface-flags-sha256'
				}
			},
			timingsMs: {
				routeLoad: 100,
				liveSessionReady: 250,
				utciCollection: 800,
				shadingCollection: 150,
				serialization: 50,
				total: 1350
			}
		});

		expect(metadata.schemaVersion).toBe('innovation-district-raw-export/v1');
		expect(metadata.analysisId).toBe('Innovation-District/2025-08-15_2m_fullday');
		expect(metadata.sourceAnalysisId).toBe('Innovation-District/innovation_district_webgpu');
		expect(metadata.sourceModelPath).toBe(RAW_SOURCE_FIELDS.sourceModelPath);
		expect(metadata.sourceGeorefPath).toBe(RAW_SOURCE_FIELDS.sourceGeorefPath);
		expect(metadata.declaredCrs).toBe('EPSG:2039');
		expect(metadata.gridSize).toBe(2);
		expect(metadata.coordinateSystem).toBe('projected-analysis');
		expect(metadata.canonicalCount).toBe(4);
		expect(metadata.activeCount).toBe(2);
		expect(metadata.hourCount).toBe(2);
		expect(metadata.hours).toEqual([0, 1]);
		expect(metadata.activeMask).toEqual({
			source: 'base+road',
			checksum: 'mask-sha256',
			signature: 'mask-signature'
		});
		expect(metadata.layout).toEqual({
			canonicalIndices: 'point-major',
			positions: 'point-major-xyz',
			utci: 'point-major-hour',
			shadingIndex: 'point-major',
			surfaceFlags: 'point-major'
		});
		expect(metadata.arrays.canonicalIndices).toEqual({
			dtype: 'u32',
			endianness: 'little',
			shape: [2],
			byteLength: canonicalIndices.byteLength
		});
		expect(metadata.arrays.positions).toEqual({
			dtype: 'f32',
			endianness: 'little',
			shape: [2, 3],
			byteLength: positions.byteLength
		});
		expect(metadata.arrays.utci).toEqual({
			dtype: 'f32',
			endianness: 'little',
			shape: [2, 2],
			byteLength: utci.byteLength
		});
		expect(metadata.arrays.shadingIndex).toEqual({
			dtype: 'f32',
			endianness: 'little',
			shape: [2],
			byteLength: shadingIndex.byteLength
		});
		expect(metadata.arrays.surfaceFlags).toEqual({
			dtype: 'u8',
			endianness: 'little',
			shape: [2],
			byteLength: surfaceFlags.byteLength
		});
		expect(metadata.files.utci?.checksum).toBe('utci-sha256');
		expect(metadata.files.surfaceFlags?.checksum).toBe('surface-flags-sha256');
		expect(metadata.timingsMs?.serialization).toBe(50);
	});

	it('throws when metadata lengths, policy fields, or byte counts do not match', () => {
		expect(() =>
			buildRawExportMetadata({
				...buildRuntimeInvalidRawMetadataParams({ schemaVersion: undefined }),
			})
		).toThrowError(/schemaVersion/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: '   ',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				}
			})
		).toThrowError(/schemaVersion/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0, 1],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([
					180600, 573600, 1.5,
					180602, 573602, 1.5
				]),
				utci: new Float32Array([30, 31, 32, 33]),
				shadingIndex: new Float32Array([0.25, 0.75]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 2,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				}
			})
		).toThrowError(/canonicalIndices/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				}
			})
		).toThrowError(/base\+road/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: '',
					signature: 'mask-signature'
				}
			})
		).toThrowError(/checksum/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: ''
				}
			})
		).toThrowError(/signature/i);

		expect(() =>
			buildRawExportMetadata(
				buildRuntimeInvalidRawMetadataParams({
					canonicalIndices: new Uint32Array([2]),
					positions: new Float32Array([180600, 573600, 1.5]),
					utci: new Float32Array([30]),
					shadingIndex: new Float32Array([0.25]),
					surfaceFlags: new Uint8Array([1]),
					activeMask: {
						source: 'base+road',
						canonicalPointCount: 1,
						checksum: 'mask-sha256',
						signature: undefined
					}
				})
			)
		).toThrowError(/signature/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				},
				files: {
					canonicalIndices: {
						fileName: '2025-08-15_2m_active-cells.canonical.u32.bin',
						checksum: ''
					}
				}
			})
		).toThrowError(/files\.canonicalIndices\.checksum/i);

		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				},
				timingsMs: {
					total: Number.NaN
				}
			})
		).toThrowError(/timingsMs\.total/i);
	});

	it('throws when coordinateSystem is not projected-analysis', () => {
		expect(() =>
			buildRawExportMetadata({
				schemaVersion: 'innovation-district-raw-export/v1',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: buildRuntimeInvalidRawMetadataParams({ coordinateSystem: 'xy_ground' })
					.coordinateSystem,
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				surfaceFlags: new Uint8Array([1]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				}
			})
		).toThrowError(/coordinateSystem/i);
	});

	it('includes raw metadata descriptors for surfaceFlags and rejects any parallel raw primary class array', () => {
		const canonicalIndices = new Uint32Array([5, 2, 9]);
		const positions = new Float32Array([
			180605, 573605, 1.5,
			180602, 573602, 1.5,
			180609, 573609, 1.5
		]);
		const utci = new Float32Array([35, 32, 40]);
		const shadingIndex = new Float32Array([0.5, 0.25, 0.75]);
		const paramsWithPrimaryClass = {
			schemaVersion: 'innovation-district-raw-export/v1',
			analysisId: 'Innovation-District/2025-08-15_2m_fullday',
			...RAW_SOURCE_FIELDS,
			coordinateSystem: 'projected-analysis' as const,
			hours: [0],
			canonicalIndices,
			positions,
			utci,
			shadingIndex,
			surfaceFlags: new Uint8Array([1, 3, 6]),
			activeMask: {
				source: 'base+road' as const,
				canonicalPointCount: 12,
				checksum: 'mask-sha256',
				signature: 'mask-signature'
			},
			files: {
				canonicalIndices: {
					fileName: '2025-08-15_2m_active-cells.canonical.u32.bin',
					checksum: 'canonical-sha256'
				},
				positions: {
					fileName: '2025-08-15_2m_active-cells.positions.f32.bin',
					checksum: 'positions-sha256'
				},
				utci: {
					fileName: '2025-08-15_2m_active-cells.utci.f32.bin',
					checksum: 'utci-sha256'
				},
				shadingIndex: {
					fileName: '2025-08-15_2m_active-cells.shading.f32.bin',
					checksum: 'shading-sha256'
				},
				surfaceFlags: {
					fileName: '2025-08-15_2m_active-cells.surface-flags.u8.bin',
					checksum: 'surface-flags-sha256'
				},
				surfaceClass: {
					fileName: '2025-08-15_2m_active-cells.surface-class.u8.bin',
					checksum: 'surface-class-sha256'
				}
			}
		};

		expect(() => buildRawExportMetadata(paramsWithPrimaryClass)).toThrowError(/surfaceClass/i);

		const params = {
			...paramsWithPrimaryClass,
			files: {
				...paramsWithPrimaryClass.files,
				surfaceClass: undefined
			}
		};
		const metadata = buildRawExportMetadata(params);
		expect(metadata.layout).toHaveProperty('surfaceFlags', 'point-major');
		expect(metadata.arrays.surfaceFlags).toEqual({
			dtype: 'u8',
			endianness: 'little',
			shape: [3],
			byteLength: 3
		});
		expect(Reflect.has(metadata.files, 'surfaceFlags')).toBe(true);
		expect(Reflect.has(metadata.files, 'surfaceClass')).toBe(false);
	});

	it('rejects classified raw metadata when a row would need to emit surface_class = unknown from zero sampled-surface flags', () => {
		const canonicalIndices = new Uint32Array([5, 2, 9]);
		const positions = new Float32Array([
			180605, 573605, 1.5,
			180602, 573602, 1.5,
			180609, 573609, 1.5
		]);
		const utci = new Float32Array([35, 32, 40]);
		const shadingIndex = new Float32Array([0.5, 0.25, 0.75]);

		const params: Parameters<typeof buildRawExportMetadata>[0] = {
			schemaVersion: 'innovation-district-raw-export/v1',
			analysisId: 'Innovation-District/2025-08-15_2m_fullday',
			...RAW_SOURCE_FIELDS,
			coordinateSystem: 'projected-analysis',
			hours: [0],
			canonicalIndices,
			positions,
			utci,
			shadingIndex,
			surfaceFlags: new Uint8Array([1, 0, 6]),
			activeMask: {
				source: 'base+road',
				canonicalPointCount: 12,
				checksum: 'mask-sha256',
				signature: 'mask-signature'
			},
			files: {
				surfaceFlags: {
					fileName: '2025-08-15_2m_active-cells.surface-flags.u8.bin',
					checksum: 'surface-flags-sha256'
				}
			}
		};

		expect(() =>
			buildRawExportMetadata(params)
		).toThrowError(/sampled-surface|surfaceFlags|unknown/i);
	});

	it('requires classified surfaceFlags for raw metadata when external input bypasses trusted typing', () => {
		const params = buildRuntimeInvalidRawMetadataParams({ surfaceFlags: undefined });

		expect(() => buildRawExportMetadata(params)).toThrowError(/surfaceFlags/i);
	});
});
