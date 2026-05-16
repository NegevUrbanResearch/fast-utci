export interface RenderStorageBufferRef {
	device: GPUDevice;
	targetBuffer: GPUBuffer;
}

export type RenderStorageReadState = {
	device?: GPUDevice;
	backendEntryAvailable: boolean;
	targetBuffer?: GPUBuffer;
};

export type RenderStorageWaitSample = {
	atMs: number;
	deviceAvailable: boolean;
	backendEntryAvailable: boolean;
	bufferAvailable: boolean;
};

export type RenderStorageWaitReadState = Omit<RenderStorageWaitSample, 'atMs'>;

export type RenderStorageWaitDiagnostics = {
	waitStartedAtMs: number;
	waitFinishedAtMs: number;
	waitMs: number;
	readAttemptCount: number;
	frameWaitCount: number;
	deviceAvailableCount: number;
	backendEntryAvailableCount: number;
	bufferAvailableCount: number;
	firstDeviceAtMs?: number;
	firstBackendEntryAtMs?: number;
	firstBufferAtMs?: number;
	lastReadState: RenderStorageWaitReadState;
	samples: RenderStorageWaitSample[];
};

export interface WaitForRenderStorageBufferParams {
	deadlineMs: number;
	now: () => number;
	waitForNextFrame: () => Promise<void>;
	isSuperseded: () => boolean;
	readStorageBuffer: () => RenderStorageBufferRef | null;
	readStorageState?: () => RenderStorageReadState;
	collectDiagnostics?: boolean;
	getTimeoutErrorMessage?: () => string;
}

export async function waitForRenderStorageBuffer(
	params: WaitForRenderStorageBufferParams
): Promise<
	RenderStorageBufferRef & { waitMs: number; waitTrace?: RenderStorageWaitDiagnostics }
> {
	const startedAt = params.now();
	const deadline = startedAt + params.deadlineMs;
	let readAttemptCount = 0;
	let frameWaitCount = 0;
	let deviceAvailableCount = 0;
	let backendEntryAvailableCount = 0;
	let bufferAvailableCount = 0;
	let firstDeviceAtMs: number | undefined;
	let firstBackendEntryAtMs: number | undefined;
	let firstBufferAtMs: number | undefined;
	let lastReadState: RenderStorageWaitReadState = {
		deviceAvailable: false,
		backendEntryAvailable: false,
		bufferAvailable: false
	};
	const firstSamples: RenderStorageWaitSample[] = [];
	const lastSamples: RenderStorageWaitSample[] = [];
	const recordSample = (sample: RenderStorageWaitSample) => {
		if (!params.collectDiagnostics) return;
		if (firstSamples.length < 4) {
			firstSamples.push(sample);
			return;
		}
		lastSamples.push(sample);
		if (lastSamples.length > 4) {
			lastSamples.shift();
		}
	};
	const buildTrace = (finishedAt: number): RenderStorageWaitDiagnostics | undefined => {
		if (!params.collectDiagnostics) return undefined;
		const lastOnlySamples = lastSamples.filter((sample) => !firstSamples.includes(sample));
		return {
			waitStartedAtMs: startedAt,
			waitFinishedAtMs: finishedAt,
			waitMs: finishedAt - startedAt,
			readAttemptCount,
			frameWaitCount,
			deviceAvailableCount,
			backendEntryAvailableCount,
			bufferAvailableCount,
			firstDeviceAtMs,
			firstBackendEntryAtMs,
			firstBufferAtMs,
			lastReadState,
			samples: [...firstSamples, ...lastOnlySamples]
		};
	};
	while (params.now() < deadline) {
		if (params.isSuperseded()) {
			throw new Error('GPU-resident render copy was superseded before storage initialization.');
		}
		const state = params.readStorageState?.();
		const storage = state
			? state.device && state.targetBuffer
				? { device: state.device, targetBuffer: state.targetBuffer }
				: null
			: params.readStorageBuffer();
		const sampleAtMs = params.now();
		const deviceAvailable = Boolean(state?.device ?? storage?.device);
		const backendEntryAvailable = state
			? state.backendEntryAvailable
			: Boolean(storage?.targetBuffer);
		const bufferAvailable = Boolean(state?.targetBuffer ?? storage?.targetBuffer);
		readAttemptCount += 1;
		if (deviceAvailable) {
			deviceAvailableCount += 1;
			firstDeviceAtMs ??= sampleAtMs;
		}
		if (backendEntryAvailable) {
			backendEntryAvailableCount += 1;
			firstBackendEntryAtMs ??= sampleAtMs;
		}
		if (bufferAvailable) {
			bufferAvailableCount += 1;
			firstBufferAtMs ??= sampleAtMs;
		}
		lastReadState = {
			deviceAvailable,
			backendEntryAvailable,
			bufferAvailable
		};
		recordSample({
			atMs: sampleAtMs,
			...lastReadState
		});
		if (storage) {
			const finishedAt = params.now();
			return {
				...storage,
				waitMs: finishedAt - startedAt,
				waitTrace: buildTrace(finishedAt)
			};
		}
		frameWaitCount += 1;
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
