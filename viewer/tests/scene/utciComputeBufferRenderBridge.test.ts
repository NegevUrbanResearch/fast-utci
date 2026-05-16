import { describe, expect, it, vi } from 'vitest';
import {
	copyComputeBufferToRenderStorage,
	waitForRenderStorageBuffer
} from '$lib/components/scene/utciComputeBufferRenderBridge';

describe('utciComputeBufferRenderBridge', () => {
	it('waits for a render storage buffer and reports timing', async () => {
		const targetBuffer = { size: 64 } as GPUBuffer;
		const result = await waitForRenderStorageBuffer({
			deadlineMs: 100,
			now: (() => {
				let time = 0;
				return () => (time += 1);
			})(),
			waitForNextFrame: async () => undefined,
			isSuperseded: () => false,
			readStorageBuffer: () => ({ device: {} as GPUDevice, targetBuffer })
		});

		expect(result.targetBuffer).toBe(targetBuffer);
		expect(result.waitMs).toBeGreaterThanOrEqual(0);
	});

	it('collects bounded storage wait diagnostics without changing wait behavior', async () => {
		const targetBuffer = { size: 64 } as GPUBuffer;
		let time = 0;
		let attempt = 0;
		const device = {} as GPUDevice;

		const result = await waitForRenderStorageBuffer({
			deadlineMs: 100,
			now: () => (time += 1),
			waitForNextFrame: async () => undefined,
			isSuperseded: () => false,
			collectDiagnostics: true,
			readStorageBuffer: () => null,
			readStorageState: () => {
				attempt += 1;
				return {
					device,
					backendEntryAvailable: attempt >= 2,
					targetBuffer: attempt >= 6 ? targetBuffer : undefined
				};
			}
		});

		expect(result.targetBuffer).toBe(targetBuffer);
		expect(result.waitTrace).toMatchObject({
			readAttemptCount: 6,
			frameWaitCount: 5,
			deviceAvailableCount: 6,
			backendEntryAvailableCount: 5,
			bufferAvailableCount: 1,
			lastReadState: {
				deviceAvailable: true,
				backendEntryAvailable: true,
				bufferAvailable: true
			}
		});
		expect(result.waitTrace?.firstDeviceAtMs).toEqual(expect.any(Number));
		expect(result.waitTrace?.firstBackendEntryAtMs).toEqual(expect.any(Number));
		expect(result.waitTrace?.firstBufferAtMs).toEqual(expect.any(Number));
		expect(result.waitTrace?.samples.length).toBeLessThanOrEqual(8);
	});

	it('fails when a copy is superseded before storage initializes', async () => {
		await expect(
			waitForRenderStorageBuffer({
				deadlineMs: 100,
				now: (() => {
					let time = 0;
					return () => (time += 1);
				})(),
				waitForNextFrame: async () => undefined,
				isSuperseded: () => true,
				readStorageBuffer: () => null
			})
		).rejects.toThrow('superseded');
	});

	it('copies compute output into render-owned storage', async () => {
		const copyBufferToBuffer = vi.fn();
		const finish = vi.fn(() => 'commands' as unknown as GPUCommandBuffer);
		const submit = vi.fn();
		const onSubmittedWorkDone = vi.fn(async () => undefined);

		await copyComputeBufferToRenderStorage({
			device: {
				createCommandEncoder: () => ({ copyBufferToBuffer, finish })
			} as unknown as GPUDevice,
			queue: { submit, onSubmittedWorkDone } as unknown as GPUQueue,
			sourceBuffer: {} as GPUBuffer,
			targetBuffer: { size: 64 } as GPUBuffer,
			byteLength: 32,
			now: performance.now.bind(performance)
		});

		expect(copyBufferToBuffer).toHaveBeenCalledWith(expect.anything(), 0, expect.anything(), 0, 32);
		expect(submit).toHaveBeenCalledWith(['commands']);
		expect(onSubmittedWorkDone).toHaveBeenCalled();
	});

	it('reports supersession after queue drain before publication', async () => {
		const copyBufferToBuffer = vi.fn();
		const finish = vi.fn(() => 'commands' as unknown as GPUCommandBuffer);
		const submit = vi.fn();
		const onSubmittedWorkDone = vi.fn(async () => undefined);

		await expect(
			copyComputeBufferToRenderStorage({
				device: {
					createCommandEncoder: () => ({ copyBufferToBuffer, finish })
				} as unknown as GPUDevice,
				queue: { submit, onSubmittedWorkDone } as unknown as GPUQueue,
				sourceBuffer: {} as GPUBuffer,
				targetBuffer: { size: 64 } as GPUBuffer,
				byteLength: 32,
				now: performance.now.bind(performance),
				isSuperseded: () => true
			})
		).rejects.toThrow('superseded');
	});
});
