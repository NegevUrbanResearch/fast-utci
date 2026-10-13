import { test, expect } from '@playwright/test';

// Test-only reproduction of the former serializer. Never an application option.
const legacy = process.env.QUALITY_LEGACY_FIRST_ROOT === '1';
test('real GPU grouped box all faces and same-size exposure clearing', async ({ page, browser }, testInfo) => {
	test.setTimeout(60000);
	await page.route('**/quality-gpu-proof.json', route => route.fulfill({
		contentType: 'application/json',
		body: '{}'
	}));
	await page.goto('/quality-gpu-proof.json');
	const result = await page.evaluate(async (legacy) => {
		const load = (path: string): Promise<any> => import(/* @vite-ignore */ path);
		const THREE = await load('/node_modules/three/build/three.module.js');
		const { serializeBvhForGpu } = await load('/src/lib/compute/gpu/bvhGpuUpload.ts');
		const { __TEST_ONLY_WebgpuUtciComputePipeline: Pipeline } = await load('/src/lib/compute/gpu/webgpuUtciPipeline.ts');
		const gpu = (navigator as any).gpu;
		if (!gpu)
			throw new Error('WebGPU unavailable');
		const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
		if (!adapter)
			throw new Error('No WebGPU adapter');
		const info = adapter.info ?? await adapter.requestAdapterInfo();
		if (info.isFallbackAdapter || adapter.isFallbackAdapter)
			throw new Error('Software fallback rejected');
		const adapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(k => [k, info[k] ?? null]));
		const limits = Object.fromEntries(['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension', 'maxComputeInvocationsPerWorkgroup'].map(k => [k, adapter.limits[k]]));
		const device = await adapter.requestDevice();
		const geometry = new THREE.BoxGeometry(2, 2, 2);
		const groupsBefore = JSON.stringify(geometry.groups);
		const indexBefore = JSON.stringify(Array.from(geometry.index.array));
		const pipeline = new Pipeline(device, false);
		const errors: string[] = [];
		device.addEventListener('uncapturederror', (e: any) => errors.push(e.error.message));
		let exposureCpuWrites = 0;
		const write = device.queue.writeBuffer.bind(device.queue);
		device.queue.writeBuffer = (buffer: any, ...args: any[]) => {
			if (buffer === pipeline.solarExposureBuffer || buffer === pipeline.skyExposureBuffer)
				exposureCpuWrites++;
			return write(buffer, ...args);
		};
		try {
			let serializedBvh = serializeBvhForGpu(geometry);
			if (legacy) {
				const { MeshBVH } = await load('/node_modules/three-mesh-bvh/src/index.js');
				const clone = geometry.clone();
				try {
					const serialized = MeshBVH.serialize(new MeshBVH(clone));
					const indexBuffer = new Uint32Array(serialized.index);
					serializedBvh = {
						bvhNodeBuffer: serialized.roots[0],
						bvhIndexBuffer: indexBuffer.buffer,
						indexBuffer,
						vertexBuffer: new Float32Array(clone.attributes.position.array)
					};
				} finally {
					clone.dispose();
				}
			}
			const sourceUnchanged = groupsBefore === JSON.stringify(geometry.groups) && indexBefore === JSON.stringify(Array.from(geometry.index.array));
			const config = {
				numPoints: 7,
				numHours: 2,
				numMonths: 1,
				diagnosticsEnabled: true
			};
			const inputs = {
				gridPoints: new Float32Array([0, 2, 0, 0, -2, 0, 2, 0, 0, -2, 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0]),
				weather: new Float32Array([30, 30, 1, 50, 800, 200, 400, 30, 30, 1, 50, 800, 200, 400]),
				domeVectors: new Float32Array([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1]),
				domeWeights: new Float32Array(6).fill(1 / 6),
				serializedBvh
			};
			const trace = () => {
				const t = pipeline.getOnDemandDiagnostics().timings.staticUploadTrace;
				return {
					clearCommandCount: t.zeroFillClearCommandCount,
					submitCount: t.zeroFillSubmitCount,
					exposureCpuWrites
				};
			};
			await pipeline.uploadStaticData({
				...inputs,
				sunVectors: new Float32Array([0, 1, 0, 0, 1, 0])
			});
			const freshTrace = trace();
			await pipeline.runExposurePrecompute(config);
			const solar = Array.from(await pipeline.readSolarExposureFull(config));
			const sky = Array.from(await pipeline.readSkyExposure(config));
			await pipeline.uploadStaticData({
				...inputs,
				sunVectors: new Float32Array(6)
			});
			const reuseTrace = trace();
			await pipeline.runExposurePrecompute(config);
			const clearedSolar = Array.from(await pipeline.readSolarExposureFull(config));
			const reusedSky = Array.from(await pipeline.readSkyExposure(config));
			await device.queue.onSubmittedWorkDone();
			return {
				adapterInfo,
				limits,
				userAgent: navigator.userAgent,
				sourceUnchanged,
				groupCount: geometry.groups.length,
				solar,
				sky,
				clearedSolar,
				reusedSky,
				freshTrace,
				reuseTrace,
				errors
			};
		} finally {
			pipeline.dispose();
			geometry.dispose();
			device.destroy();
		}
	}, legacy);
	const evidence = {
		browserVersion: browser.version(),
		legacy,
		...result
	};
	await testInfo.attach('real-gpu-exposure-proof', {
		body: JSON.stringify(evidence, null, 2),
		contentType: 'application/json'
	});
	expect(result.errors).toEqual([]);
	expect(result.sourceUnchanged).toBe(true);
	expect(result.groupCount).toBe(6);
	expect(result.solar).toEqual([1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
	for (const value of result.sky.slice(0, 6))
		expect(Math.abs((value as number) - 5 / 6)).toBeLessThan(1e-6);
	expect(result.sky[6]).toBe(0);
	expect(result.clearedSolar).toEqual(new Array(14).fill(0));
	expect(result.reusedSky).toEqual(result.sky);
	expect(result.freshTrace).toEqual({
		clearCommandCount: 0,
		submitCount: 0,
		exposureCpuWrites: 0
	});
	expect(result.reuseTrace).toEqual({
		clearCommandCount: 2,
		submitCount: 1,
		exposureCpuWrites: 0
	});
});
