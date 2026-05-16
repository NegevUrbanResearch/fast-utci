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

	it('counts hover samples only after suppression and throttle gates', () => {
		if (!existsSync(tooltipLayerPath)) return;

		const source = readFileSync(tooltipLayerPath, 'utf8');
		const handlerSource = source.slice(source.indexOf('function handleMouseMove'));
		const suppressIndex = handlerSource.indexOf('if (hoverPolicy.shouldSuppress)');
		const throttleIndex = handlerSource.indexOf('if (hoverPolicy.shouldThrottle)');
		const incrementIndex = handlerSource.indexOf('tooltipHoverSampleCount += 1;');

		expect(suppressIndex).toBeGreaterThanOrEqual(0);
		expect(throttleIndex).toBeGreaterThanOrEqual(0);
		expect(incrementIndex).toBeGreaterThanOrEqual(0);
		expect(incrementIndex).toBeGreaterThan(suppressIndex);
		expect(incrementIndex).toBeGreaterThan(throttleIndex);
	});
});
