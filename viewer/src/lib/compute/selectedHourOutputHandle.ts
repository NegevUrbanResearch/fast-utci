export type SelectedHourOutputSource = 'webgpu-on-demand-snapshot';

export interface SelectedHourOutputHandle {
	readonly buffer: GPUBuffer;
	readonly byteLength: number;
	requestId?: number;
	timeIndex?: number;
	readonly source: SelectedHourOutputSource;
	disposed: boolean;
	dispose(): void;
}

export interface SelectedHourOutputHandleParams {
	buffer: GPUBuffer;
	byteLength: number;
	requestId?: number;
	timeIndex?: number;
	source: SelectedHourOutputSource;
}

export function createSelectedHourOutputHandle(
	params: SelectedHourOutputHandleParams
): SelectedHourOutputHandle {
	const handle: SelectedHourOutputHandle = {
		buffer: params.buffer,
		byteLength: params.byteLength,
		requestId: params.requestId,
		timeIndex: params.timeIndex,
		source: params.source,
		disposed: false,
		dispose() {
			if (handle.disposed) return;
			handle.buffer.destroy();
			handle.disposed = true;
		}
	};
	return handle;
}

export function disposeSelectedHourOutputHandle(
	handle: SelectedHourOutputHandle | null | undefined
): void {
	handle?.dispose();
}
