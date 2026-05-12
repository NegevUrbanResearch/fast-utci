import { describe, it, expect, expectTypeOf } from 'vitest';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
	loadReferenceIntermediatesFromFs,
	type MrtReference,
	type SkyReference,
	type SolarReference
} from '$lib/parity/loadReferenceIntermediatesFromFs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');
const TEST_DIR = resolve(fileURLToPath(import.meta.url), '..');

describe('loadReferenceIntermediatesFromFs', () => {
	it('rejects when solar file does not exist', async () => {
		const basePath = resolve(REPO_ROOT, 'data/analyses/nonexistent_intermediates');
		await expect(loadReferenceIntermediatesFromFs(basePath, 'solar')).rejects.toThrow();
	});

	it('rejects when sky file does not exist', async () => {
		const basePath = resolve(REPO_ROOT, 'data/analyses/nonexistent_sky');
		await expect(loadReferenceIntermediatesFromFs(basePath, 'sky')).rejects.toThrow();
	});

	it('when solar file exists, returns numPositions, numHours, solarExposure Float32Array', async () => {
		const basePath = resolve(TEST_DIR, 'fixtures/ben_gurion');
		const ref = await loadReferenceIntermediatesFromFs(basePath, 'solar');
		expectTypeOf(ref).toEqualTypeOf<SolarReference>();
		expect(ref).toHaveProperty('numPositions');
		expect(ref).toHaveProperty('numHours');
		expect(ref).toHaveProperty('solarExposure');
		expect(ref.numPositions).toBe(2);
		expect(ref.numHours).toBe(24);
		expect(ref.solarExposure).toBeInstanceOf(Float32Array);
		expect(ref.solarExposure.length).toBe(ref.numPositions * ref.numHours);
	});

	it('when sky file exists, returns numPositions, skyExposure Float32Array', async () => {
		const basePath = resolve(TEST_DIR, 'fixtures/ben_gurion');
		const ref = await loadReferenceIntermediatesFromFs(basePath, 'sky');
		expectTypeOf(ref).toEqualTypeOf<SkyReference>();
		expect(ref).toHaveProperty('numPositions');
		expect(ref).toHaveProperty('skyExposure');
		expect(ref.numPositions).toBe(2);
		expect(ref.skyExposure).toBeInstanceOf(Float32Array);
		expect(ref.skyExposure.length).toBe(ref.numPositions);
	});

	it('narrows mrt stage calls to MrtReference at compile time', () => {
		const basePath = resolve(TEST_DIR, 'fixtures/ben_gurion');
		const loadMrt = () => loadReferenceIntermediatesFromFs(basePath, 'mrt');
		expectTypeOf(loadMrt).returns.toEqualTypeOf<Promise<MrtReference>>();
	});
});
