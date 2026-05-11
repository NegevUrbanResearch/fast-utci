import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(__dirname, '../..');
const tooltipLayerPath = resolve(
	repoRoot,
	'src/routes/main/MainRouteTooltipLayer.svelte'
);

describe('main route tooltip layer source lock', () => {
	it('reads the comparison mesh through the event-time getter inside pointer handling', () => {
		if (!existsSync(tooltipLayerPath)) return;

		const source = readFileSync(tooltipLayerPath, 'utf8');
		expect(source).toMatch(/export let getComparisonUtciMesh/);
		expect(source).toMatch(/function handleMouseMove/);
		const handlerSource = source.slice(source.indexOf('function handleMouseMove'));
		expect(handlerSource).toMatch(/getComparisonUtciMesh\(\)/);
	});
});
