export interface RenderStorageBufferRef {
	device: GPUDevice;
	targetBuffer: GPUBuffer;
}

export interface WaitForRenderStorageBufferParams {
	deadlineMs: number;
	now: () => number;
	waitForNextFrame: () => Promise<void>;
	isSuperseded: () => boolean;
	readStorageBuffer: () => RenderStorageBufferRef | null;
	getTimeoutErrorMessage?: () => string;
}

export async function waitForRenderStorageBuffer(
	params: WaitForRenderStorageBufferParams
): Promise<RenderStorageBufferRef & { waitMs: number }> {
	const startedAt = params.now();
	const deadline = startedAt + params.deadlineMs;
	while (params.now() < deadline) {
		if (params.isSuperseded()) {
			throw new Error('GPU-resident render copy was superseded before storage initialization.');
		}
		const storage = params.readStorageBuffer();
		if (storage) {
			return { ...storage, waitMs: params.now() - startedAt };
		}
		await params.waitForNextFrame();
	}
	throw new Error(
		params.getTimeoutErrorMessage?.() ?? 'Timed out waiting for render-owned UTCI storage buffer.'
	);
}

export interface CopyComputeBufferToRenderStorageParams {
	device: GPUDevice;
	queue: GPUQueue;
	sourceBuffer: GPUBuffer;
	targetBuffer: GPUBuffer & { size?: number };
	byteLength: number;
	now: () => number;
	isSuperseded?: () => boolean;
}

export async function copyComputeBufferToRenderStorage(
	params: CopyComputeBufferToRenderStorageParams
): Promise<{ bufferCopyMs: number; queueDrainMs: number }> {
	if (params.targetBuffer.size !== undefined && params.targetBuffer.size < params.byteLength) {
		throw new Error('Three storage buffer is smaller than the accepted compute output buffer.');
	}
	const copyStartedAt = params.now();
	const encoder = params.device.createCommandEncoder();
	encoder.copyBufferToBuffer(params.sourceBuffer, 0, params.targetBuffer, 0, params.byteLength);
	params.queue.submit([encoder.finish()]);
	const bufferCopyMs = params.now() - copyStartedAt;
	const queueDrainStartedAt = params.now();
	await params.queue.onSubmittedWorkDone();
	if (params.isSuperseded?.()) {
		throw new Error('GPU-resident render copy was superseded after queue drain.');
	}
	return {
		bufferCopyMs,
		queueDrainMs: params.now() - queueDrainStartedAt
	};
}
