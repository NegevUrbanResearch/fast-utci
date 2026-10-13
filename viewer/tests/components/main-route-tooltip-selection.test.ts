import { describe, expect, it, vi } from 'vitest';
import { render, fireEvent, waitFor } from '@testing-library/svelte/svelte5';
import * as THREE from 'three';
import MainRouteTooltipLayer from '../../src/routes/main/MainRouteTooltipLayer.svelte';
import { createSelectedHourOutputHandle } from '$lib/compute/gpu/selectedHourOutputHandle';
import { getTooltipDataAsync } from '$lib/services/tooltipService';

vi.mock('$lib/services/tooltipService', async (importOriginal) => ({
	...await importOriginal<typeof import('$lib/services/tooltipService')>(),
	getTooltipDataAsync: vi.fn()
}));

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((done) => {
		resolve = done;
	});
	return {
		promise,
		resolve
	};
}

describe('main route pending tooltip selection', () => {
	it.each(['metricType', 'viewerCurrentHour', 'requestedMonthIndex', 'baseDisplayedAnalysis', 'basePendingGpuResidentOutput'] as const)('rejects delayed hover on %s change and permits the next hover', async (changedProp) => {
		const pending = deferred<any>();
		const read = vi.mocked(getTooltipDataAsync).mockReset();
		read.mockReturnValueOnce(pending.promise).mockResolvedValue({
			value: 0.65,
			positionIndex: 0,
			position: {
				x: 0,
				y: 0,
				z: 0
			}
		});
		const canvas = document.createElement('canvas');
		document.body.append(canvas);
		const analysis = {
			metadata: { analysis_type: 'single_hour' },
			data: {
				numPositions: 1,
				numHours: 1,
				positions: new Float32Array([0, 0, 0])
			}
		} as any;
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as any,
			byteLength: 4,
			timeIndex: 12,
			source: 'webgpu-on-demand-snapshot'
		});
		const accepted = {
			requestId: 1,
			monthIndex: 7,
			timeIndex: 12,
			hourIndex: 12,
			metricType: 'utci',
			gpuOutputHandle: handle
		} as any;
		const props = {
			canvasElement: canvas,
			cameraRef: new THREE.PerspectiveCamera(),
			baseMesh: new THREE.Mesh(),
			baseDisplayedAnalysis: analysis,
			baseSceneTimeIndex: 12,
			basePendingGpuResidentOutput: accepted,
			comparisonDisplayedAnalysis: null,
			comparisonSceneTimeIndex: undefined,
			rendererDevice: {} as GPUDevice,
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			mainViewportElement: null,
			curtainPosition: 50,
			viewerCurrentHour: 12,
			requestedMonthIndex: 7,
			metricType: 'utci' as const,
			utciVisible: true
		};
		const { container, rerender, unmount } = render(MainRouteTooltipLayer, { props });
		await fireEvent.mouseMove(canvas, {
			clientX: 20,
			clientY: 20
		});
		await waitFor(() => expect(read).toHaveBeenCalledTimes(1));
		const changes = {
			metricType: 'shading_index',
			viewerCurrentHour: 13,
			requestedMonthIndex: 8,
			baseDisplayedAnalysis: { ...analysis },
			basePendingGpuResidentOutput: { requestId: 2 }
		};
		await rerender({ [changedProp]: changes[changedProp] });
		pending.resolve({
			value: 29,
			positionIndex: 0,
			position: {
				x: 0,
				y: 0,
				z: 0
			}
		});
		await pending.promise;
		await new Promise((done) => setTimeout(done, 0));
		expect(container.querySelector('[role="tooltip"]')).toBeNull();
		expect(handle.disposed).toBe(false);
		await new Promise((done) => setTimeout(done, 20));
		await fireEvent.mouseMove(canvas, {
			clientX: 21,
			clientY: 21
		});
		await waitFor(() => expect(container.querySelector('[role="tooltip"]')).not.toBeNull());
		expect(container.textContent).not.toContain('29.0');
		unmount();
		canvas.remove();
	});
	it('keeps monthly Shading Index point reads valid when an irrelevant stale hour changes', async () => {
		const pending = deferred<any>();
		const read = vi.mocked(getTooltipDataAsync).mockReset().mockReturnValue(pending.promise);
		const canvas = document.createElement('canvas');
		document.body.append(canvas);
		const analysis = {
			metadata: { analysis_type: 'single_hour' },
			data: {
				numPositions: 1,
				numHours: 1,
				positions: new Float32Array([0, 0, 0])
			}
		} as any;
		const handle = createSelectedHourOutputHandle({
			buffer: { destroy: vi.fn() } as any,
			byteLength: 4,
			source: 'webgpu-on-demand-snapshot',
			metricType: 'shading_index',
			period: {
				kind: 'month-index',
				index: 7,
				startTimeIndex: 168,
				timeCount: 24
			}
		});
		const accepted = {
			requestId: 1,
			monthIndex: 7,
			timeIndex: 168,
			hourIndex: 0,
			metricType: 'shading_index',
			gpuOutputHandle: handle
		} as any;
		const props = {
			canvasElement: canvas,
			cameraRef: new THREE.PerspectiveCamera(),
			baseMesh: new THREE.Mesh(),
			baseDisplayedAnalysis: analysis,
			baseSceneTimeIndex: 168,
			basePendingGpuResidentOutput: accepted,
			comparisonDisplayedAnalysis: null,
			comparisonSceneTimeIndex: undefined,
			rendererDevice: {} as GPUDevice,
			useLiveUtciOnMainRoute: true,
			isComparing: false,
			mainViewportElement: null,
			curtainPosition: 50,
			viewerCurrentHour: 0,
			requestedMonthIndex: 7,
			metricType: 'shading_index' as const,
			utciVisible: true
		};
		const { container, rerender, unmount } = render(MainRouteTooltipLayer, { props });
		await fireEvent.mouseMove(canvas, {
			clientX: 20,
			clientY: 20
		});
		await waitFor(() => expect(read).toHaveBeenCalledOnce());
		expect(read.mock.calls[0][7]?.metricPointValueReader).toMatchObject({
			monthIndex: 7,
			snapshotIdentity: handle,
			readbackByteLength: 4
		});
		await rerender({ viewerCurrentHour: 17 });
		pending.resolve({
			value: 0.65,
			positionIndex: 0,
			position: {
				x: 0,
				y: 0,
				z: 0
			}
		});
		await waitFor(() => expect(container.querySelector('[role="tooltip"]')).not.toBeNull());
		expect(container.textContent).toContain('Shading Index');
		expect(container.textContent).toContain('0.65');
		expect(handle.disposed).toBe(false);
		unmount();
		canvas.remove();
	});
});
