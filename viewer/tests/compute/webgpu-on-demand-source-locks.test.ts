import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

describe('WebGPU on-demand source guards', () => {
	const testDir = dirname(fileURLToPath(import.meta.url));
	const source = readFileSync(
		resolve(testDir, '../../src/lib/compute/gpu/webgpuUtciPipeline.ts'),
		'utf8'
	).replace(/\r\n/g, '\n');
	const renderBridgeSource = readFileSync(
		resolve(testDir, '../../src/lib/services/gpuUtciRenderBridge.ts'),
		'utf8'
	).replace(/\r\n/g, '\n');
	const onDemandShaderSource = readFileSync(
		resolve(testDir, '../../src/lib/compute/gpu/shaders/mrt_utci_on_demand.wgsl'),
		'utf8'
	);

	function getSection(startMarker: string, endMarker: string): string {
		const start = source.indexOf(startMarker);
		if (start === -1) return '';

		const end = source.indexOf(endMarker, start);
		return source.slice(start, end === -1 ? undefined : end);
	}

	function extractFunctionBody(sourceText: string, functionName: string): string {
		const declarationStart = sourceText.indexOf(`function ${functionName}`);
		expect(declarationStart).toBeGreaterThanOrEqual(0);

		const bodyStart = sourceText.indexOf('{', declarationStart);
		expect(bodyStart).toBeGreaterThanOrEqual(0);

		let depth = 0;
		for (let index = bodyStart; index < sourceText.length; index += 1) {
			const character = sourceText[index];
			if (character === '{') {
				depth += 1;
			} else if (character === '}') {
				depth -= 1;
				if (depth === 0) {
					return sourceText.slice(bodyStart + 1, index);
				}
			}
		}

		throw new Error(`Unable to extract ${functionName} body`);
	}

	const exposureHelperSource = getSection('private async encodeExposurePasses', '\n\n\tasync runAll');
	const clearBvhStateSource = getSection('private clearBvhState()', '\n\n\tprivate assertMatchesLastConfig');
	const assertMatchesLastConfigSource = getSection(
		'private assertMatchesLastConfig',
		'\n\n\tasync uploadStaticData'
	);
	const uploadStaticDataSource = getSection('async uploadStaticData(', 'private createBvhBindGroup(');
	const runAllMethodSource = getSection('async runAll', '\n\n\tasync runExposurePrecompute');
	const exposureMethodSource = getSection('async runExposurePrecompute', '\n\n\tasync runUtciForTimeIndex');
	const onDemandMethodSource = getSection('async runUtciForTimeIndex', '\n\n\tasync readOnDemandUtciForDebug');
	const readUtcisSliceSource = getSection('async readUtcisSlice', '\n\n\tasync readUtciBulk');
	const readUtciBulkSource = getSection('async readUtciBulk', '\n\n\tasync readSolarExposureFull');
	const readSolarExposureSource = getSection('async readSolarExposureFull', '\n\n\tasync readSkyExposure');
	const readSkyExposureSource = getSection('async readSkyExposure', '\n\n\tasync readMrtFull');

	it('keeps current all-hours production path available', () => {
		expect(runAllMethodSource.includes('async runAll')).toBe(true);
		expect(runAllMethodSource.includes('this.ensurePipeline()')).toBe(true);
		expect(runAllMethodSource.includes('this.utciBuffer')).toBe(true);
		expect(runAllMethodSource.includes('this.mrtBuffer')).toBe(true);
		expect(runAllMethodSource.includes('await this.encodeExposurePasses')).toBe(true);
		expect(source.includes('readUtciBulk')).toBe(true);
	});

	it('refreshes persistent exposure allocation tracking during baseline runs too', () => {
		expect(runAllMethodSource.includes('persistentExposureBytes')).toBe(true);
		expect(runAllMethodSource.includes('solarExposureBytes + skyExposureBytes')).toBe(true);
		expect(runAllMethodSource.includes('allHoursOutputBytes: utciBytes + mrtBytes')).toBe(true);
	});

	it('adds exposure-only precompute separate from runAll', () => {
		expect(exposureMethodSource.includes('async runExposurePrecompute')).toBe(true);
		expect(exposureMethodSource.includes('await this.encodeExposurePasses')).toBe(true);
		expect(exposureMethodSource.includes('this.ensureSolarPipeline()')).toBe(true);
		expect(exposureMethodSource.includes('this.ensureSkyPipeline()')).toBe(true);
		expect(exposureMethodSource.includes('await this.ensureWeatherBuffer()')).toBe(true);
		expect(exposureMethodSource.includes('this.ensurePipeline()')).toBe(false);
		expect(exposureMethodSource.includes('utciBytes = numPoints * totalTimeSteps * 4')).toBe(false);
		expect(exposureMethodSource.includes('this.utciBuffer')).toBe(false);
		expect(exposureMethodSource.includes('this.mrtBuffer')).toBe(false);
		expect(exposureMethodSource.includes('this.shortErfBuffer')).toBe(false);
		expect(exposureMethodSource.includes('this.longErfBuffer')).toBe(false);
		expect(exposureMethodSource.includes('this.shortDmrtBuffer')).toBe(false);
		expect(exposureMethodSource.includes('this.longDmrtBuffer')).toBe(false);
	});

	it('keeps the extracted exposure helper free of UTCI and MRT allocation work', () => {
		expect(exposureHelperSource.includes('private async encodeExposurePasses')).toBe(true);
		expect(exposureHelperSource.includes('solarPipeline')).toBe(true);
		expect(exposureHelperSource.includes('skyPipeline')).toBe(true);
		expect(exposureHelperSource.includes('this.utciBuffer')).toBe(false);
		expect(exposureHelperSource.includes('this.mrtBuffer')).toBe(false);
		expect(exposureHelperSource.includes('this.shortErfBuffer')).toBe(false);
		expect(exposureHelperSource.includes('this.longErfBuffer')).toBe(false);
		expect(exposureHelperSource.includes('this.shortDmrtBuffer')).toBe(false);
		expect(exposureHelperSource.includes('this.longDmrtBuffer')).toBe(false);
	});

	it('invalidates prior exposure-run state when new static data is uploaded', () => {
		expect(uploadStaticDataSource.includes('async uploadStaticData')).toBe(true);
		expect(uploadStaticDataSource.includes('this.ranExposurePassesThisRun = false')).toBe(true);
	});

	it('clears stale BVH state and snapshots uploaded weather', () => {
		expect(uploadStaticDataSource.includes('this.weatherData = new Float32Array(params.weather)')).toBe(true);
		expect(uploadStaticDataSource.includes('this.clearBvhState();')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhNodeBuffer?.destroy();')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhNodeBuffer = null;')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhIndexBuffer?.destroy();')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhIndexBuffer = null;')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhVertexBuffer?.destroy();')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhVertexBuffer = null;')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhParamsBuffer?.destroy();')).toBe(true);
		expect(clearBvhStateSource.includes('this.bvhParamsBuffer = null;')).toBe(true);
	});

	it('allows exposure readback after either supported exposure entrypoint', () => {
		expect(readSolarExposureSource.includes('run runAll or runExposurePrecompute first')).toBe(true);
		expect(readSkyExposureSource.includes('run runAll or runExposurePrecompute first')).toBe(true);
	});

	it('validates readback dimensions against the producing run config', () => {
		expect(readUtcisSliceSource.includes('this.assertMatchesLastConfig')).toBe(true);
		expect(readUtciBulkSource.includes('this.assertMatchesLastConfig')).toBe(true);
		expect(assertMatchesLastConfigSource.includes('this.lastConfig.numPoints !== numPoints')).toBe(true);
		expect(assertMatchesLastConfigSource.includes('this.lastConfig.numHours !== numHours')).toBe(true);
		expect(assertMatchesLastConfigSource.includes('this.lastConfig.numMonths !== numMonths')).toBe(true);
		expect(assertMatchesLastConfigSource.includes('does not match the last run config')).toBe(true);
	});

	it('locks the on-demand prototype source entrypoints and shader contract', () => {
		expect(source.includes('ensureOnDemandPipeline')).toBe(true);
		expect(source.includes('runUtciForTimeIndex')).toBe(true);
		expect(source.includes('readOnDemandUtciForDebug')).toBe(true);
		expect(onDemandMethodSource.includes('this.assertMatchesLastConfig')).toBe(true);
		expect(onDemandMethodSource.includes("context: 'runUtciForTimeIndex'")).toBe(true);
		expect(onDemandMethodSource.includes('await this.ensureWeatherBuffer()')).toBe(true);
		expect(onDemandMethodSource.includes('this.ranExposurePassesThisRun')).toBe(true);
		expect(onDemandMethodSource.includes('run runAll first')).toBe(false);
		expect(onDemandShaderSource.includes('time_index')).toBe(true);
		expect(onDemandShaderSource.includes('output_format')).toBe(true);
		expect(onDemandShaderSource.includes('output_utci')).toBe(true);
	});

	it('records tracked GPU allocation bytes without using browser total VRAM APIs', () => {
		expect(source.includes('mergeTrackedGpuAllocationBytes')).toBe(true);
		expect(source.includes('trackedGpuAllocationBytes')).toBe(true);
		expect(source.includes('performance.memory')).toBe(false);
		expect(source.includes('measureUserAgentSpecificMemory')).toBe(false);
	});

	it('keeps the UTCI render bridge honest about cpu-uploaded versus compute-buffer sources', () => {
		expect(renderBridgeSource.includes('export type GpuNativeUtciSurfaceSource')).toBe(true);
		expect(renderBridgeSource.includes("'cpu-uploaded-selected-hour'")).toBe(true);
		expect(renderBridgeSource.includes("'compute-buffer-selected-hour'")).toBe(true);
		expect(renderBridgeSource.includes("source: 'cpu-uploaded-selected-hour'")).toBe(true);
		expect(renderBridgeSource.includes("source: 'compute-buffer-selected-hour'")).toBe(true);
		expect(renderBridgeSource.includes('export function createComputeBufferUtciSurfaceMesh')).toBe(
			true
		);
		expect(renderBridgeSource.includes('export function updateComputeBufferUtciSurfaceMesh')).toBe(
			true
		);
		expect(renderBridgeSource.includes('pendingComputeBufferUtciSource')).toBe(true);
		expect(renderBridgeSource.includes('createVertexToPointIndexArray')).toBe(true);
	});

	it('derives compute-buffer cell lookup from local surface coordinates', () => {
		const colorNodeBody = extractFunctionBody(renderBridgeSource, 'createUtciColorNode');
		expect(colorNodeBody).toContain('positionLocal.x');
		expect(colorNodeBody).toContain('positionLocal.z');
		expect(colorNodeBody).toContain('.floor()');
		expect(colorNodeBody).toContain('clamp(');
		expect(colorNodeBody).toContain('row.mul(uint(layout.width)).add(column)');
		expect(colorNodeBody).toContain('cellToPointStorage.element(cellIndex)');
		expect(colorNodeBody).toContain('utciStorage.element(pointIndex)');
		expect(colorNodeBody).not.toContain(
			'vertexIndex.div(uint(SURFACE_VERTICES_PER_CELL))'
		);
		expect(renderBridgeSource).toContain('createIndexedGridSurfaceGeometry');

		const computeBufferBody = extractFunctionBody(
			renderBridgeSource,
			'createComputeBufferUtciSurfaceMesh'
		);
		expect(computeBufferBody).toContain('createUtciColorNode(');
		expect(computeBufferBody).toMatch(/createUtciColorNode\(\s*options\.layout,/);
	});
});
