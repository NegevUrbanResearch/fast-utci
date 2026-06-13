import type { SelectedHourRenderStorageCopyPreflight } from '$lib/diagnostics/selectedHourRenderPublicationDiagnostics';

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

const GPU_BUFFER_COPY_SRC = globalThis.GPUBufferUsage?.COPY_SRC ?? 4;
const GPU_BUFFER_COPY_DST = globalThis.GPUBufferUsage?.COPY_DST ?? 8;
const GPU_BUFFER_STORAGE = globalThis.GPUBufferUsage?.STORAGE ?? 128;

function hasUsageFlag(usage: number | undefined, flag: number): boolean | undefined {
	return usage === undefined ? undefined : (usage & flag) === flag;
}

export class RenderStorageCopyPreflightError extends Error {
	constructor(readonly copyPreflight: SelectedHourRenderStorageCopyPreflight) {
		super(
			`Render storage copy preflight failed: ${copyPreflight.failureReasons?.join('; ')}.`
		);
		this.name = 'RenderStorageCopyPreflightError';
	}
}

export function buildRenderStorageCopyPreflight(
	params: CopyComputeBufferToRenderStorageParams
): SelectedHourRenderStorageCopyPreflight {
	const sourceWithDiagnostics = params.sourceBuffer as GPUBuffer & {
		size?: number;
		usage?: number;
	};
	const targetWithDiagnostics = params.targetBuffer as GPUBuffer & {
		size?: number;
		usage?: number;
	};
	const sourceByteLength = sourceWithDiagnostics.size ?? params.byteLength;
	const targetByteLength = targetWithDiagnostics.size;
	const sourceHasCopySrcUsage = hasUsageFlag(
		sourceWithDiagnostics.usage,
		GPU_BUFFER_COPY_SRC
	);
	const targetHasCopyDstUsage = hasUsageFlag(
		targetWithDiagnostics.usage,
		GPU_BUFFER_COPY_DST
	);
	const targetHasStorageUsage = hasUsageFlag(
		targetWithDiagnostics.usage,
		GPU_BUFFER_STORAGE
	);
	const failureReasons: string[] = [];
	if (sourceWithDiagnostics.size === undefined) {
		failureReasons.push('source buffer size is unknown');
	}
	if (targetByteLength === undefined) {
		failureReasons.push('target buffer size is unknown');
	}
	if (!Number.isFinite(params.byteLength) || params.byteLength < 0) {
		failureReasons.push('requested copy byteLength is invalid');
	}
	if (
		sourceWithDiagnostics.size !== undefined &&
		sourceWithDiagnostics.size !== params.byteLength
	) {
		failureReasons.push(
			'source buffer size does not match requested copy byteLength'
		);
	}
	if (targetByteLength !== undefined && targetByteLength !== params.byteLength) {
		failureReasons.push(
			'target buffer size does not match requested copy byteLength'
		);
	}
	if (
		sourceWithDiagnostics.size !== undefined &&
		targetByteLength !== undefined &&
		targetByteLength !== sourceByteLength
	) {
		failureReasons.push('source/target byte lengths differ');
	}
	if (sourceHasCopySrcUsage === undefined) {
		failureReasons.push('source buffer usage is unknown');
	}
	if (
		targetHasCopyDstUsage === undefined ||
		targetHasStorageUsage === undefined
	) {
		failureReasons.push('target buffer usage is unknown');
	}
	if (sourceHasCopySrcUsage === false) {
		failureReasons.push('source buffer is missing COPY_SRC usage');
	}
	if (targetHasCopyDstUsage === false) {
		failureReasons.push('target buffer is missing COPY_DST usage');
	}
	if (targetHasStorageUsage === false) {
		failureReasons.push('target buffer is missing STORAGE usage');
	}

	return {
		status: failureReasons.length === 0 ? 'passed' : 'failed',
		sourceByteLength,
		targetByteLength,
		requestedByteLength: params.byteLength,
		byteLengthsMatch:
			sourceWithDiagnostics.size !== undefined &&
			targetByteLength !== undefined &&
			sourceByteLength === params.byteLength &&
			targetByteLength === params.byteLength,
		sourceUsage: sourceWithDiagnostics.usage,
		targetUsage: targetWithDiagnostics.usage,
		sourceHasCopySrcUsage,
		targetHasCopyDstUsage,
		targetHasStorageUsage,
		failureReasons: failureReasons.length > 0 ? failureReasons : undefined
	};
}

export async function copyComputeBufferToRenderStorage(
	params: CopyComputeBufferToRenderStorageParams
): Promise<{
	bufferCopyMs: number;
	queueDrainMs: number;
	queueDrainStartedAtMs: number;
	queueDrainCompletedAtMs: number;
	copyEncoderCreateMs: number;
	copyCommandRecordMs: number;
	copySubmitMs: number;
	copyPreflight: SelectedHourRenderStorageCopyPreflight;
}> {
	const copyPreflight = buildRenderStorageCopyPreflight(params);
	if (copyPreflight.status === 'failed') {
		throw new RenderStorageCopyPreflightError(copyPreflight);
	}
	const copyStartedAt = params.now();
	const encoderStartedAt = params.now();
	const encoder = params.device.createCommandEncoder();
	const copyEncoderCreateMs = params.now() - encoderStartedAt;
	const commandRecordStartedAt = params.now();
	encoder.copyBufferToBuffer(params.sourceBuffer, 0, params.targetBuffer, 0, params.byteLength);
	const commandBuffer = encoder.finish();
	const copyCommandRecordMs = params.now() - commandRecordStartedAt;
	const submitStartedAt = params.now();
	params.queue.submit([commandBuffer]);
	const copySubmitMs = params.now() - submitStartedAt;
	const bufferCopyMs = params.now() - copyStartedAt;
	const queueDrainStartedAtMs = params.now();
	await params.queue.onSubmittedWorkDone();
	const queueDrainCompletedAtMs = params.now();
	if (params.isSuperseded?.()) {
		throw new Error('GPU-resident render copy was superseded after queue drain.');
	}
	return {
		bufferCopyMs,
		queueDrainMs: queueDrainCompletedAtMs - queueDrainStartedAtMs,
		queueDrainStartedAtMs,
		queueDrainCompletedAtMs,
		copyEncoderCreateMs,
		copyCommandRecordMs,
		copySubmitMs,
		copyPreflight
	};
}
