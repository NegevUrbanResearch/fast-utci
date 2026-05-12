export type WebgpuLargeBufferRequiredLimits = {
	maxStorageBufferBindingSize: number;
	maxBufferSize: number;
};

export type WebgpuLargeBufferDeviceLimits = Partial<WebgpuLargeBufferRequiredLimits>;

export const LARGE_BUFFER_REQUIRED_LIMITS: WebgpuLargeBufferRequiredLimits = {
	maxStorageBufferBindingSize: 512 * 1024 * 1024,
	maxBufferSize: 1024 * 1024 * 1024
};

export function createLargeBufferRequiredLimits(): WebgpuLargeBufferRequiredLimits {
	return { ...LARGE_BUFFER_REQUIRED_LIMITS };
}

export function readLargeBufferDeviceLimits(
	device: GPUDevice | undefined
): WebgpuLargeBufferDeviceLimits | undefined {
	if (!device) return undefined;
	return {
		maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
		maxBufferSize: device.limits.maxBufferSize
	};
}
