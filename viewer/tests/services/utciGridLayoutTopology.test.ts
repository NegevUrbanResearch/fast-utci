import { describe, expect, it } from 'vitest';
import { buildUtciGridLayout } from '$lib/services/pointCloudService';
import {
	getActiveMaskGridCell,
	getActiveMaskPointGridCell
} from '$lib/services/utciGridLayoutTopology';
import type { Analysis } from '$lib/types/analysis';

function createActiveMaskAnalysis(params: {
	coordinateSystem: 'xy_ground' | 'xz_ground';
	activeCanonicalIndices: number[];
	width: number;
	height: number;
}): Analysis {
	const pointCount = params.activeCanonicalIndices.length;
	return {
		metadata: {
			analysis_type: 'single_hour',
			num_positions: pointCount,
			hours: ['00:00'],
			utci_range: { min: 10, max: 40 },
			grid_size: 1,
			coordinate_system: params.coordinateSystem,
			model_file: 'active-mask.glb',
			bounds: {
				x_min: 0,
				x_max: params.width - 1,
				y_min: 0,
				y_max: params.height - 1,
				z: 0
			},
			activeMask: {
				source: 'base',
				canonicalPointCount: params.width * params.height,
				activePointCount: pointCount,
				inactivePointCount: params.width * params.height - pointCount,
				activePointRatio: pointCount / (params.width * params.height),
				activeMaskChecksum: 'active-mask-test',
				signature: 'active-mask-signature',
				activeCanonicalIndices: new Uint32Array(params.activeCanonicalIndices)
			}
		},
		data: {
			numPositions: pointCount,
			numHours: 1,
			positions: new Float32Array(pointCount * 3),
			utciValues: new Float32Array(pointCount)
		}
	};
}

describe('UTCI grid layout topology', () => {
	it('builds compact active topology without dense render-only arrays', () => {
		const analysis = createActiveMaskAnalysis({
			coordinateSystem: 'xz_ground',
			width: 4,
			height: 3,
			activeCanonicalIndices: [0, 5, 11]
		});

		const layout = buildUtciGridLayout(analysis);

		expect(layout.renderTopology).toBe('active-cells');
		if (layout.renderTopology !== 'active-cells') {
			throw new Error(`Expected active-cells layout, received ${layout.renderTopology}.`);
		}
		expect(layout.renderCellCount).toBe(3);
		expect(layout.canonicalCellCount).toBe(12);
		expect(layout.numPositions).toBe(3);
		expect(layout.width).toBe(4);
		expect(layout.height).toBe(3);
		expect(layout).not.toHaveProperty('cellToPointIndex');
		expect(layout).not.toHaveProperty('colorBuffer');
		expect(layout).not.toHaveProperty('indexToTexel');
		expect(layout).not.toHaveProperty('indexToColumn');
		expect(layout).not.toHaveProperty('indexToRow');
		expect(layout).toMatchObject({
			activeMaskSignature: 'active-mask-signature'
		});
		expect(Array.from(layout.activeCanonicalIndices)).toEqual([0, 5, 11]);
		expect([0, 1, 2].map((pointIndex) => getActiveMaskPointGridCell({ layout, pointIndex }))).toEqual([
			getActiveMaskGridCell({
				canonicalIndex: 0,
				width: 4,
				height: 3,
				coordinateSystem: 'xz_ground'
			}),
			getActiveMaskGridCell({
				canonicalIndex: 5,
				width: 4,
				height: 3,
				coordinateSystem: 'xz_ground'
			}),
			getActiveMaskGridCell({
				canonicalIndex: 11,
				width: 4,
				height: 3,
				coordinateSystem: 'xz_ground'
			})
		]);
	});

	it('derives xy_ground active rows from canonical index without dense precompute', () => {
		const analysis = createActiveMaskAnalysis({
			coordinateSystem: 'xy_ground',
			width: 3,
			height: 3,
			activeCanonicalIndices: [0, 1, 5, 8]
		});

		const layout = buildUtciGridLayout(analysis);

		expect(layout.renderTopology).toBe('active-cells');
		if (layout.renderTopology !== 'active-cells') {
			throw new Error(`Expected active-cells layout, received ${layout.renderTopology}.`);
		}
		expect(layout.renderCellCount).toBe(4);
		expect(layout.canonicalCellCount).toBe(9);
		expect(layout).not.toHaveProperty('cellToPointIndex');
		expect(layout).not.toHaveProperty('colorBuffer');
		expect(layout).not.toHaveProperty('indexToTexel');
		expect(layout).not.toHaveProperty('indexToColumn');
		expect(layout).not.toHaveProperty('indexToRow');
		expect([0, 1, 2, 3].map((pointIndex) => getActiveMaskPointGridCell({ layout, pointIndex }))).toEqual([
			getActiveMaskGridCell({
				canonicalIndex: 0,
				width: 3,
				height: 3,
				coordinateSystem: 'xy_ground'
			}),
			getActiveMaskGridCell({
				canonicalIndex: 1,
				width: 3,
				height: 3,
				coordinateSystem: 'xy_ground'
			}),
			getActiveMaskGridCell({
				canonicalIndex: 5,
				width: 3,
				height: 3,
				coordinateSystem: 'xy_ground'
			}),
			getActiveMaskGridCell({
				canonicalIndex: 8,
				width: 3,
				height: 3,
				coordinateSystem: 'xy_ground'
			})
		]);
	});
});
