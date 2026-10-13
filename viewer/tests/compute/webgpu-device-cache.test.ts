import { afterEach, expect, it, vi } from 'vitest';
import { createWebgpuUtciPipeline, __resetWebgpuDeviceCacheForTests } from '$lib/compute/gpu/webgpuUtciPipeline';

afterEach(() => {
	vi.unstubAllGlobals();
	__resetWebgpuDeviceCacheForTests();
});

it('shares a rejected acquisition and allows a later retry', async () => {
	const device = {
		limits: { maxStorageBuffersPerShaderStage: 8 },
		queue: {},
		lost: new Promise(() => { })
	};
	const requestDevice = vi.fn().mockRejectedValueOnce(new Error('transient')).mockResolvedValue(device);
	const requestAdapter = vi.fn().mockResolvedValue({
		limits: device.limits,
		requestDevice
	});
	vi.stubGlobal('navigator', { gpu: { requestAdapter } });
	const results = await Promise.allSettled([createWebgpuUtciPipeline(), createWebgpuUtciPipeline()]);
	expect(results.map((result) => result.status)).toEqual(['rejected', 'rejected']);
	expect(requestDevice).toHaveBeenCalledTimes(1);
	expect((await createWebgpuUtciPipeline()).getDeviceForDebug?.()).toBe(device);
	expect(requestDevice).toHaveBeenCalledTimes(2);
});

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: unknown) => void;
	const promise = new Promise<T>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return {
		promise,
		resolve,
		reject
	};
}

it('keeps metadata bound to overlapping normal and diagnostics acquisitions', async () => {
	const normal = deferred<any>();
	const diagnostics = deferred<any>();
	const lossA = deferred<any>();
	const lossB = deferred<any>();
	const deviceA = {
		limits: { maxStorageBuffersPerShaderStage: 8 },
		queue: {},
		lost: lossA.promise
	};
	const deviceB = {
		limits: { maxStorageBuffersPerShaderStage: 10 },
		queue: {},
		lost: lossB.promise
	};
	const requestDevice = vi.fn().mockReturnValueOnce(normal.promise).mockReturnValueOnce(diagnostics.promise).mockResolvedValue(deviceB);
	const requestAdapter = vi.fn().mockResolvedValue({
		limits: deviceB.limits,
		requestDevice
	});
	vi.stubGlobal('navigator', { gpu: { requestAdapter } });
	const first = createWebgpuUtciPipeline();
	const second = createWebgpuUtciPipeline({ enableDiagnostics: true });
	const third = createWebgpuUtciPipeline({ enableDiagnostics: true });
	diagnostics.resolve(deviceB);
	const [b, c] = await Promise.all([second, third]);
	normal.resolve(deviceA);
	const a = await first;
	expect(a.getDeviceForDebug?.()).toBe(deviceA);
	expect(a.supportsMrtComponentDiagnostics?.()).toBe(false);
	expect(b.getDeviceForDebug?.()).toBe(deviceB);
	expect(c.getDeviceForDebug?.()).toBe(deviceB);
	expect(b.supportsMrtComponentDiagnostics?.()).toBe(true);
	expect(requestDevice).toHaveBeenCalledTimes(2);
	lossA.resolve({});
	await Promise.resolve();
	expect((await createWebgpuUtciPipeline()).getDeviceForDebug?.()).toBe(deviceB);
	expect(requestDevice).toHaveBeenCalledTimes(2);
	lossB.resolve({});
	await Promise.resolve();
	await createWebgpuUtciPipeline();
	expect(requestDevice).toHaveBeenCalledTimes(3);
});

it('does not let an older rejection clear a newer diagnostics acquisition', async () => {
	const normal = deferred<any>();
	const device = {
		limits: { maxStorageBuffersPerShaderStage: 10 },
		queue: {},
		lost: new Promise(() => { })
	};
	const requestDevice = vi.fn().mockReturnValueOnce(normal.promise).mockResolvedValue(device);
	vi.stubGlobal('navigator', {
		gpu: {
			requestAdapter: vi.fn().mockResolvedValue({
				limits: device.limits,
				requestDevice
			})
		}
	});
	const first = createWebgpuUtciPipeline();
	const rejection = expect(first).rejects.toThrow('old failure');
	await createWebgpuUtciPipeline({ enableDiagnostics: true });
	normal.reject(new Error('old failure'));
	await rejection;
	expect((await createWebgpuUtciPipeline()).getDeviceForDebug?.()).toBe(device);
	expect(requestDevice).toHaveBeenCalledTimes(2);
});

it('provided devices bypass acquisition and remain externally owned', async () => {
	const requestAdapter = vi.fn();
	vi.stubGlobal('navigator', { gpu: { requestAdapter } });
	const device = {
		limits: { maxStorageBuffersPerShaderStage: 8 },
		queue: {},
		destroy: vi.fn()
	};
	const pipeline = await createWebgpuUtciPipeline({ device: device as any });
	pipeline.dispose?.();
	expect(requestAdapter).not.toHaveBeenCalled();
	expect(device.destroy).not.toHaveBeenCalled();
});
