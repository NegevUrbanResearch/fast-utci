import { describe, it, expect } from 'vitest';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadWebgpuCollectedFromFs } from '$lib/parity/loadWebgpuCollectedFromFs';

const TEST_DIR = resolve(fileURLToPath(import.meta.url), '..');
const fixtureBase = resolve(TEST_DIR, 'fixtures/empty');

describe('loadWebgpuCollectedFromFs', () => {
	it('returns solar when _webgpu_solar.json exists', async () => {
		const out = await loadWebgpuCollectedFromFs(fixtureBase);
		expect(out.solar).toBeDefined();
		expect(out.solar?.numPositions).toBe(2);
		expect(out.solar?.numHours).toBe(24);
		expect(out.solar?.solarExposure).toBeDefined();
		const solarExposure = out.solar?.solarExposure;
		expect(Array.isArray(solarExposure) || ArrayBuffer.isView(solarExposure as unknown)).toBe(true);
	});
});
