import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(__dirname, '../..');
const viewportPath = 'src/routes/main/MainRouteViewport.svelte';

function readIfPresent(relativePath: string): string | null {
	const absolutePath = resolve(repoRoot, relativePath);
	if (!existsSync(absolutePath)) return null;
	return readFileSync(absolutePath, 'utf8');
}

describe('main route viewport source lock', () => {
	it('keeps the viewport component product-only when it exists', () => {
		const source = readIfPresent(viewportPath);
		if (source == null) return;

		expect(source).toMatch(/<Scene/);
		expect(source).toMatch(/<Model/);
		expect(source).toMatch(/<UTCIPointCloud/);
		expect(source).toMatch(/<ComparisonRenderer/);
		expect(source).not.toMatch(/\.bin/i);
		expect(source).not.toMatch(/\bparity\b/i);
		expect(source).not.toMatch(/Python/i);
		expect(source).not.toMatch(/__onDemandPrototypeDiagnostics__/i);
	});
});
