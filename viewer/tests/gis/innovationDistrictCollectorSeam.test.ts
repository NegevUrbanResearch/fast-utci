import { describe, expect, it, vi } from 'vitest';

import {
	FAST_UTCI_COLLECTOR_EXPORT_QUERY_FLAG,
	buildFastUtciCollectorExport,
	installFastUtciCollectorExport,
	type FastUtciCollectorExportWindow
} from '../../src/routes/main/collectorExportSeam';
import type { Analysis } from '$lib/types/analysis';

function createAnalysis(
	activeMask?: Analysis['metadata']['activeMask'],
	positions = new Float32Array(0)
): Analysis {
	return {
		metadata: {
			analysis_type: 'full_day',
			num_positions: activeMask?.activePointCount ?? 4,
			hours: ['00:00', '01:00'],
			utci_range: { min: 20, max: 40 },
			grid_size: 2,
			coordinate_system: 'xy_ground',
			model_file: 'data/3d_models/Innovation-District/innovation_district.glb',
			source_analysis_id: 'Innovation-District/innovation_district_webgpu',
			bounds: {
				x_min: 10,
				x_max: 12,
				y_min: 20,
				y_max: 22,
				z: 1.5
			},
			activeMask
		},
		data: {
			numPositions: activeMask?.activePointCount ?? 4,
			numHours: 2,
			positions,
			utciByHour: []
		}
	};
}

function createActiveMask(
	overrides: Partial<NonNullable<Analysis['metadata']['activeMask']>> = {}
): NonNullable<Analysis['metadata']['activeMask']> {
	const activeCanonicalIndices = new Uint32Array([0, 3]);
	return {
		source: 'base+road',
		canonicalPointCount: 4,
		activePointCount: 2,
		inactivePointCount: 2,
		activePointRatio: 0.5,
		activeMaskChecksum: 'mask-sha256',
		activeCanonicalIndices,
		signature: 'mask-signature',
		...overrides
	};
}

function createOutput(metricType: 'utci' | 'shading_index') {
	return {
		requestId: 12,
		metricType,
		monthIndex: 7,
		hourIndex: metricType === 'utci' ? 3 : 0,
		timeIndex: metricType === 'utci' ? 171 : 168,
		output: {
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-on-demand-snapshot',
			metricType,
			valueLayout: 'one-f32-per-point',
			period:
				metricType === 'utci'
					? { kind: 'time-index', index: 171 }
					: { kind: 'month-index', index: 7, startTimeIndex: 168, timeCount: 24 },
			numPoints: 2,
			outputBytes: 8
		},
		gpuOutputHandle: {
			buffer: {} as GPUBuffer,
			byteLength: 8,
			source: 'webgpu-on-demand-snapshot',
			ownerId: 'webgpu-on-demand-snapshot',
			metricType,
			valueLayout: 'one-f32-per-point',
			period:
				metricType === 'utci'
					? { kind: 'time-index', index: 171 }
					: { kind: 'month-index', index: 7, startTimeIndex: 168, timeCount: 24 },
			disposed: false,
			dispose: vi.fn()
		},
		utciRange: { min: 20, max: 40 }
	};
}

describe('collector export seam lifecycle', () => {
	it('does not expose the collector export unless the query flag is present', () => {
		const win = {} as FastUtciCollectorExportWindow;

		installFastUtciCollectorExport({
			win,
			searchParams: new URLSearchParams('utciRenderDiagnostics=1'),
			getCurrent: () => null
		});

		expect(win.__fastUtciCollectorExport).toBeUndefined();

		installFastUtciCollectorExport({
			win,
			searchParams: new URLSearchParams(`${FAST_UTCI_COLLECTOR_EXPORT_QUERY_FLAG}=1`),
			getCurrent: () => null
		});

		expect(win.__fastUtciCollectorExport).toEqual(expect.any(Function));
	});

	it('removes the collector export on teardown', () => {
		const win = {} as FastUtciCollectorExportWindow;
		const cleanup = installFastUtciCollectorExport({
			win,
			searchParams: new URLSearchParams(`${FAST_UTCI_COLLECTOR_EXPORT_QUERY_FLAG}=1`),
			getCurrent: () => null
		});

		expect(win.__fastUtciCollectorExport).toEqual(expect.any(Function));
		cleanup();
		expect(win.__fastUtciCollectorExport).toBeUndefined();
	});
});

describe('buildFastUtciCollectorExport', () => {
	it('requires active canonical indices and expected active mask source', async () => {
		await expect(
			buildFastUtciCollectorExport({
				analysis: createAnalysis(createActiveMask({ activeCanonicalIndices: undefined as never })),
				output: createOutput('utci') as never,
				device: {} as GPUDevice,
				readF32MetricOutput: vi.fn().mockResolvedValue(new Float32Array([30, 31]))
			})
		).rejects.toThrowError(/activeCanonicalIndices/i);

		await expect(
			buildFastUtciCollectorExport({
				analysis: createAnalysis(createActiveMask({ source: 'base' })),
				output: createOutput('utci') as never,
				device: {} as GPUDevice,
				readF32MetricOutput: vi.fn().mockResolvedValue(new Float32Array([30, 31]))
			})
		).rejects.toThrowError(/base\+road/i);
	});

	it('refuses rectangular-only or canonical-only analysis data', async () => {
		await expect(
			buildFastUtciCollectorExport({
				analysis: createAnalysis(undefined),
				output: createOutput('utci') as never,
				device: {} as GPUDevice,
				readF32MetricOutput: vi.fn().mockResolvedValue(new Float32Array([30, 31]))
			})
		).rejects.toThrowError(/activeMask/i);

		await expect(
			buildFastUtciCollectorExport({
				analysis: createAnalysis(
					createActiveMask({
						activePointCount: 4,
						inactivePointCount: 0,
						activePointRatio: 1,
						activeCanonicalIndices: new Uint32Array([0, 1, 2, 3])
					})
				),
				output: createOutput('utci') as never,
				device: {} as GPUDevice,
				readF32MetricOutput: vi.fn().mockResolvedValue(new Float32Array([30, 31, 32, 33]))
			})
		).rejects.toThrowError(/rectangular-only|canonical-only/i);
	});

	it('exports compact active rows with active mask metadata', async () => {
		const result = await buildFastUtciCollectorExport({
			analysis: createAnalysis(
				createActiveMask(),
				new Float32Array([
					101.25, 202.5, 3.75,
					104.5, 205.25, 3.75
				])
			),
			output: createOutput('utci') as never,
			device: {} as GPUDevice,
			readF32MetricOutput: vi.fn().mockResolvedValue(new Float32Array([30, 31]))
		});

		expect(result.metadata.activeMask.activeCanonicalIndices).toEqual(new Uint32Array([0, 3]));
		expect(result.metadata.activeMask.source).toBe('base+road');
		expect(result.metadata.metricType).toBe('utci');
		expect(result.values).toEqual(new Float32Array([30, 31]));
		expect(result.positions.length).toBe(6);
	});

	it('preserves supplied live active-row Analysis.data.positions exactly', async () => {
		const livePositions = new Float32Array([
			999.125, 888.25, 7.5,
			777.5, 666.75, 8.25
		]);

		const result = await buildFastUtciCollectorExport({
			analysis: createAnalysis(createActiveMask(), livePositions),
			output: createOutput('utci') as never,
			device: {} as GPUDevice,
			readF32MetricOutput: vi.fn().mockResolvedValue(new Float32Array([30, 31]))
		});

		expect(result.positions).toEqual(livePositions);
		expect(result.positions).not.toBe(livePositions);
	});
});
