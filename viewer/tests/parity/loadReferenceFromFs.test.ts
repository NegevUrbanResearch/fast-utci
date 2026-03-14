import { describe, it, expect } from 'vitest';
import { resolve } from 'node:path';
import { loadReferenceFromFs } from '$lib/parity/loadReferenceFromFs';

const REPO_ROOT = resolve(process.cwd(), process.cwd().endsWith('viewer') ? '..' : '.');

describe('loadReferenceFromFs', () => {
	it('loads Ben-Gurion base .bin + .json and returns positions and utciByHour', async () => {
		const basePath = resolve(REPO_ROOT, 'data/analyses/Ben-Gurion/20250815_grid_2m_fullday');
		const ref = await loadReferenceFromFs(basePath);
		expect(ref.metadata.num_positions).toBe(ref.data.numPositions);
		expect(ref.data.positions.length).toBe(ref.data.numPositions * 3);
		expect(ref.data.utciByHour.length).toBe(24);
		expect(ref.data.utciByHour[0].length).toBe(ref.data.numPositions);
	});

	it('throws when .bin does not exist', async () => {
		await expect(
			loadReferenceFromFs(resolve(REPO_ROOT, 'data/analyses/nonexistent'))
		).rejects.toThrow();
	});
});
