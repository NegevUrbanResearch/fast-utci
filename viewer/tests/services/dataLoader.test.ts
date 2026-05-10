import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
	parseSingleHourBinary,
	parseFullDayBinary,
	calculateStatistics,
	getUTCIValue,
	getPosition,
	getUTCIForHour,
	loadAnalysisMetadataOnly
} from '$lib/services/dataLoader';
import { buildUtciGridLayout } from '$lib/services/pointCloudService';
import type { SingleHourData, FullDayData } from '$lib/types/analysis';

// Helper to create ArrayBuffer with binary data
function createSingleHourBinary(numPositions: number, positions: number[], utciValues: number[]): ArrayBuffer {
	const buffer = new ArrayBuffer(4 + numPositions * 12 + numPositions * 4);
	const view = new DataView(buffer);
	let offset = 0;
	
	// Write numPositions (uint32)
	view.setUint32(offset, numPositions, true);
	offset += 4;
	
	// Write positions (float32 x, y, z)
	for (let i = 0; i < numPositions * 3; i++) {
		view.setFloat32(offset, positions[i], true);
		offset += 4;
	}
	
	// Write UTCI values (float32)
	for (let i = 0; i < numPositions; i++) {
		view.setFloat32(offset, utciValues[i], true);
		offset += 4;
	}
	
	return buffer;
}

function createFullDayBinary(numPositions: number, numHours: number, positions: number[], utciByHour: number[][]): ArrayBuffer {
	const buffer = new ArrayBuffer(8 + numPositions * 12 + numHours * numPositions * 4);
	const view = new DataView(buffer);
	let offset = 0;
	
	// Write header (numPositions, numHours)
	view.setUint32(offset, numPositions, true);
	offset += 4;
	view.setUint32(offset, numHours, true);
	offset += 4;
	
	// Write positions
	for (let i = 0; i < numPositions * 3; i++) {
		view.setFloat32(offset, positions[i], true);
		offset += 4;
	}
	
	// Write UTCI values hour by hour
	for (let hour = 0; hour < numHours; hour++) {
		for (let i = 0; i < numPositions; i++) {
			view.setFloat32(offset, utciByHour[hour][i], true);
			offset += 4;
		}
	}
	
	return buffer;
}

describe('dataLoader service', () => {
	describe('loadAnalysisMetadataOnly', () => {
		beforeEach(() => {
			vi.restoreAllMocks();
		});

		it('loads metadata and derives positions without fetching binary UTCI data', async () => {
			const requestedUrls: string[] = [];
			vi.stubGlobal(
				'fetch',
				vi.fn(async (url: string) => {
					requestedUrls.push(url);
					return {
						ok: true,
						json: async () => ({
							analysis_type: 'full_day',
							num_positions: 4,
							hours: ['00:00', '01:00'],
							utci_range: { min: 10, max: 40 },
							grid_size: 1,
							coordinate_system: 'xz_ground',
							model_file: 'model.glb',
							bounds: { x_min: 0, x_max: 1, y_min: 0, y_max: 1, z: 0 }
						})
					};
				})
			);

			const analysis = await loadAnalysisMetadataOnly('Project/analysis');

			expect(requestedUrls).toEqual(['/data/analyses/Project/analysis.json']);
			expect(requestedUrls.some((url) => url.includes('.bin'))).toBe(false);
			expect(analysis.metadata.source_analysis_id).toBe('Project/analysis');
			expect(analysis.data.numPositions).toBe(4);
			expect(analysis.data.positions).toEqual(
				new Float32Array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1])
			);
			expect(analysis.data.utciByHour).toEqual([]);
		});

		it('stores xy-ground metadata-only grid positions in analysis coordinates', async () => {
			vi.stubGlobal(
				'fetch',
				vi.fn(async () => ({
					ok: true,
					json: async () => ({
						analysis_type: 'full_day',
						num_positions: 4,
						hours: ['00:00'],
						utci_range: { min: 10, max: 40 },
						grid_size: 1,
						coordinate_system: 'xy_ground',
						model_file: 'model.glb',
						bounds: { x_min: 0, x_max: 1, y_min: 10, y_max: 11, z: 2 }
					})
				}))
			);

			const analysis = await loadAnalysisMetadataOnly('Project/xy-analysis');

			expect(analysis.data.positions).toEqual(
				new Float32Array([0, 10, 2, 0, 11, 2, 1, 10, 2, 1, 11, 2])
			);
		});

		it('builds a two-dimensional xy-ground surface layout from metadata-only positions', async () => {
			vi.stubGlobal(
				'fetch',
				vi.fn(async () => ({
					ok: true,
					json: async () => ({
						analysis_type: 'full_day',
						num_positions: 4,
						hours: ['00:00'],
						utci_range: { min: 10, max: 40 },
						grid_size: 1,
						coordinate_system: 'xy_ground',
						model_file: 'model.glb',
						bounds: { x_min: 0, x_max: 1, y_min: 10, y_max: 11, z: 2 }
					})
				}))
			);

			const analysis = await loadAnalysisMetadataOnly('Project/xy-layout');
			const layout = buildUtciGridLayout(analysis);

			expect(layout.width).toBe(2);
			expect(layout.height).toBe(2);
			expect(layout.centerX).toBeCloseTo(0);
			expect(layout.centerZ).toBeCloseTo(0);
		});

		it('rejects metadata-derived positions when the grid count disagrees with metadata', async () => {
			vi.stubGlobal(
				'fetch',
				vi.fn(async () => ({
					ok: true,
					json: async () => ({
						analysis_type: 'full_day',
						num_positions: 3,
						hours: ['00:00'],
						utci_range: { min: 10, max: 40 },
						grid_size: 1,
						coordinate_system: 'xz_ground',
						model_file: 'model.glb',
						bounds: { x_min: 0, x_max: 1, y_min: 0, y_max: 1, z: 0 }
					})
				}))
			);

			await expect(loadAnalysisMetadataOnly('Project/mismatch')).rejects.toThrow(
				/does not match metadata num_positions/
			);
		});
	});

	describe('parseSingleHourBinary', () => {
		it('should parse single hour binary data', () => {
			const numPositions = 2;
			const positions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 positions x 3 coords
			const utciValues = [20.5, 25.0];
			
			const buffer = createSingleHourBinary(numPositions, positions, utciValues);
			const result = parseSingleHourBinary(buffer);
			
			expect(result.numPositions).toBe(2);
			expect(result.numHours).toBe(1);
			expect(result.positions).toBeInstanceOf(Float32Array);
			expect(result.utciValues).toBeInstanceOf(Float32Array);
			expect(result.positions.length).toBe(6);
			expect(result.utciValues.length).toBe(2);
		});

		it('should correctly read positions', () => {
			const positions = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
			const utciValues = [20.0, 25.0];
			const buffer = createSingleHourBinary(2, positions, utciValues);
			const result = parseSingleHourBinary(buffer);
			
			expect(result.positions[0]).toBeCloseTo(10.0, 5);
			expect(result.positions[3]).toBeCloseTo(40.0, 5);
		});

		it('should correctly read UTCI values', () => {
			const positions = [0, 0, 0, 0, 0, 0];
			const utciValues = [15.5, 22.3];
			const buffer = createSingleHourBinary(2, positions, utciValues);
			const result = parseSingleHourBinary(buffer);
			
			expect(result.utciValues[0]).toBeCloseTo(15.5, 5);
			expect(result.utciValues[1]).toBeCloseTo(22.3, 5);
		});
	});

	describe('parseFullDayBinary', () => {
		it('should parse full day binary data', () => {
			const numPositions = 2;
			const numHours = 3;
			const positions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
			const utciByHour = [
				[20.0, 21.0],
				[22.0, 23.0],
				[24.0, 25.0]
			];
			
			const buffer = createFullDayBinary(numPositions, numHours, positions, utciByHour);
			const result = parseFullDayBinary(buffer);
			
			expect(result.numPositions).toBe(2);
			expect(result.numHours).toBe(3);
			expect(result.positions).toBeInstanceOf(Float32Array);
			expect(result.utciByHour).toHaveLength(3);
			expect(result.utciByHour[0]).toBeInstanceOf(Float32Array);
			expect(result.utciByHour[0].length).toBe(2);
		});
	});

	describe('calculateStatistics', () => {
		it('should calculate min, max, and mean', () => {
			const values = new Float32Array([10, 20, 30, 40, 50]);
			const stats = calculateStatistics(values);
			
			expect(stats.min).toBe(10);
			expect(stats.max).toBe(50);
			expect(stats.mean).toBe(30);
			expect(stats.count).toBe(5);
		});

		it('should handle NaN and Infinity values', () => {
			const values = new Float32Array([10, NaN, 20, Infinity, 30]);
			const stats = calculateStatistics(values);
			
			expect(stats.min).toBe(10);
			expect(stats.max).toBe(30);
			expect(stats.mean).toBe(20);
			expect(stats.count).toBe(3);
		});

		it('should handle empty array', () => {
			const values = new Float32Array([]);
			const stats = calculateStatistics(values);
			
			expect(stats.min).toBe(Infinity);
			expect(stats.max).toBe(-Infinity);
			expect(stats.mean).toBe(0);
			expect(stats.count).toBe(0);
		});
	});

	describe('getUTCIValue', () => {
		it('should get UTCI value for single hour data', () => {
			const data: SingleHourData = {
				numPositions: 2,
				numHours: 1,
				positions: new Float32Array([0, 0, 0, 1, 1, 1]),
				utciValues: new Float32Array([20.0, 25.0])
			};
			
			const value = getUTCIValue(data, 0);
			expect(value).toBe(20.0);
			
			const value2 = getUTCIValue(data, 1);
			expect(value2).toBe(25.0);
		});

		it('should get UTCI value for full day data', () => {
			const data: FullDayData = {
				numPositions: 2,
				numHours: 2,
				positions: new Float32Array([0, 0, 0, 1, 1, 1]),
				utciByHour: [
					new Float32Array([20.0, 21.0]),
					new Float32Array([22.0, 23.0])
				]
			};
			
			const value = getUTCIValue(data, 0, 0);
			expect(value).toBe(20.0);
			
			const value2 = getUTCIValue(data, 1, 1);
			expect(value2).toBe(23.0);
		});
	});

	describe('getPosition', () => {
		it('should get position coordinates', () => {
			const data: SingleHourData = {
				numPositions: 2,
				numHours: 1,
				positions: new Float32Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
				utciValues: new Float32Array([20.0, 25.0])
			};
			
			const pos0 = getPosition(data, 0);
			expect(pos0.x).toBe(10.0);
			expect(pos0.y).toBe(20.0);
			expect(pos0.z).toBe(30.0);
			
			const pos1 = getPosition(data, 1);
			expect(pos1.x).toBe(40.0);
			expect(pos1.y).toBe(50.0);
			expect(pos1.z).toBe(60.0);
		});
	});

	describe('getUTCIForHour', () => {
		it('should get UTCI values for single hour', () => {
			const data: SingleHourData = {
				numPositions: 2,
				numHours: 1,
				positions: new Float32Array([0, 0, 0, 1, 1, 1]),
				utciValues: new Float32Array([20.0, 25.0])
			};
			
			const values = getUTCIForHour(data);
			expect(values).toBe(data.utciValues);
		});

		it('should get UTCI values for specific hour in full day', () => {
			const data: FullDayData = {
				numPositions: 2,
				numHours: 2,
				positions: new Float32Array([0, 0, 0, 1, 1, 1]),
				utciByHour: [
					new Float32Array([20.0, 21.0]),
					new Float32Array([22.0, 23.0])
				]
			};
			
			const hour0 = getUTCIForHour(data, 0);
			expect(hour0).toBe(data.utciByHour[0]);
			
			const hour1 = getUTCIForHour(data, 1);
			expect(hour1).toBe(data.utciByHour[1]);
		});
	});
});


