import { describe, expect, it } from 'vitest';
import { join, resolve } from 'node:path';

import {
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
const dataOutDirArg = join('data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw');
const parentRelativeDataOutDirArg = join('..', 'data', 'gis', 'Innovation-District', '2025-08-15_2m', 'raw');
const relativeOutDirArg = join('tmp-gis-smoke');

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
			shadingIndex: '2025-08-15_2m_active-cells.shading.f32.bin'
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
			hours: [0, 1],
			activeMaskSource: 'base+road'
		});

		expect(Array.from(result.canonicalIndices)).toEqual([2, 5]);
		expect(Array.from(result.positions)).toEqual([
			180600, 573600, 1.5, 180602, 573602, 1.5
		]);
		expect(Array.from(result.utci)).toEqual([30, 31, 32, 33]);
		expect(Array.from(result.shadingIndex)).toEqual([0.25, 0.75]);
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
				hours: [0]
			})
		).toThrowError(/activeMaskSource/i);

		expect(() =>
			buildActiveCellArrays({
				activeCanonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utciByHour: [new Float32Array([30])],
				shadingIndex: new Float32Array([0.25]),
				hours: [0],
				activeMaskSource: 'base'
			})
		).toThrowError(/base\+road/i);
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
			shadingIndex: 'point-major'
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
		expect(metadata.files.utci?.checksum).toBe('utci-sha256');
		expect(metadata.timingsMs?.serialization).toBe(50);
	});

	it('throws when metadata lengths, policy fields, or byte counts do not match', () => {
		expect(() =>
			buildRawExportMetadata({
				schemaVersion: undefined as unknown as string,
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
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
				schemaVersion: '   ',
				analysisId: 'Innovation-District/2025-08-15_2m_fullday',
				...RAW_SOURCE_FIELDS,
				coordinateSystem: 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
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
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: ''
				}
			})
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
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: undefined as unknown as string
				}
			})
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
				coordinateSystem: 'xy_ground' as unknown as 'projected-analysis',
				hours: [0],
				canonicalIndices: new Uint32Array([2]),
				positions: new Float32Array([180600, 573600, 1.5]),
				utci: new Float32Array([30]),
				shadingIndex: new Float32Array([0.25]),
				activeMask: {
					source: 'base+road',
					canonicalPointCount: 1,
					checksum: 'mask-sha256',
					signature: 'mask-signature'
				}
			})
		).toThrowError(/coordinateSystem/i);
	});
});
