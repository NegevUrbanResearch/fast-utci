// @vitest-environment node
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { readFile, realpath } from 'node:fs/promises';
import { GET } from '../../src/routes/data/[...path]/+server';

vi.mock('node:fs/promises', async (importOriginal) => {
	const fs = await importOriginal<typeof import('node:fs/promises')>();
	return { ...fs, readFile: vi.fn(fs.readFile), realpath: vi.fn(fs.realpath) };
});

let fixture: string;
let dataRoot: string;
const bytes = new Uint8Array([0, 1, 127, 255]);
const request = (path: string) => GET({ params: { path } } as Parameters<typeof GET>[0]);

beforeAll(async () => {
	fixture = await mkdtemp(join(tmpdir(), 'utci-data-route-'));
	dataRoot = join(fixture, 'data');
	await mkdir(join(fixture, 'viewer'));
	for (const folder of ['models/nested', 'weather/nested', 'analyses/nested']) {
		await mkdir(join(dataRoot, folder), { recursive: true });
	}
	for (const filename of ['models/nested/model.glb', 'models/nested/model.gltf', 'weather/nested/weather.json', 'analyses/nested/result.bin', 'models/nested/%2e%2e.json']) {
		await writeFile(join(dataRoot, filename), bytes);
	}
	await writeFile(join(fixture, 'outside.json'), bytes);
	await mkdir(join(fixture, 'data-sibling'));
	await writeFile(join(fixture, 'data-sibling/outside.json'), bytes);
});

beforeEach(() => {
	vi.spyOn(process, 'cwd').mockReturnValue(join(fixture, 'viewer'));
	vi.spyOn(console, 'error').mockImplementation(() => {});
	vi.mocked(readFile).mockClear();
	vi.mocked(realpath).mockClear();
});

afterEach(() => vi.restoreAllMocks());
afterAll(async () => {
	if (resolve(fixture).startsWith(resolve(tmpdir()) + '\\') || resolve(fixture).startsWith(resolve(tmpdir()) + '/')) {
		await rm(fixture, { recursive: true, force: true });
	}
});

describe('local data route', () => {
	it.each([
		['models/nested/model.glb', 'model/gltf-binary'],
		['models/nested/model.gltf', 'model/gltf+json'],
		['weather/nested/weather.json', 'application/json'],
		['analyses/nested/result.bin', 'application/octet-stream'],
		['models/nested/%2e%2e.json', 'application/json']
	])('serves %s unchanged without decoding filenames again', async (path, contentType) => {
		const response = await request(path);
		expect(response.status).toBe(200);
		expect(response.headers.get('Content-Type')).toBe(contentType);
		expect(response.headers.get('Access-Control-Allow-Origin')).toBe('*');
		expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
	});

	it('returns 400 for an empty path', async () => {
		expect((await request('')).status).toBe(400);
		expect(readFile).not.toHaveBeenCalled();
	});

	it('returns 404 for a missing contained file', async () => {
		expect((await request('models/missing.glb')).status).toBe(404);
	});

	it.each(['../outside.json', '..\\outside.json', '../data-sibling/outside.json', '.', 'models/..', '/outside.json', '\\outside.json', 'C:\\outside.json', 'C:outside.json', '//server/share/outside.json', 'models/invalid\0.json'])('rejects unsafe path %s before reading', async (path) => {
		expect((await request(path)).status).toBe(400);
		expect(readFile).not.toHaveBeenCalled();
	});

	it('rejects decoded route traversal before reading', async () => {
		expect((await request(decodeURIComponent('%2e%2e%2foutside.json'))).status).toBe(400);
		expect(readFile).not.toHaveBeenCalled();
	});

	it('rejects an in-root symlink whose real target escapes before reading', async () => {
		// Windows symlink privileges are not required: mock only filesystem canonicalization.
		vi.mocked(realpath).mockImplementation(async (path) =>
			String(path) === dataRoot ? dataRoot : join(fixture, 'outside.json')
		);
		expect((await request('models/nested/link.json')).status).toBe(400);
		expect(readFile).not.toHaveBeenCalled();
	});
});
