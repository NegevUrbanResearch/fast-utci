import { describe, it, expect } from 'vitest';
import { getNormalizedMousePosition } from '$lib/utils/mouse';

describe('mouse utilities', () => {
	it('should normalize mouse position to -1 to 1 range', () => {
		const rect: DOMRect = {
			left: 0,
			top: 0,
			width: 100,
			height: 100,
			x: 0,
			y: 0,
			bottom: 100,
			right: 100,
			toJSON: () => ({})
		};
		const event = { clientX: 50, clientY: 50 } as MouseEvent;
		
		const result = getNormalizedMousePosition(event, rect);
		
		expect(result.x).toBeCloseTo(0, 2);
		expect(result.y).toBeCloseTo(0, 2);
	});

	it('should normalize top-left corner to (-1, 1)', () => {
		const rect: DOMRect = {
			left: 0,
			top: 0,
			width: 100,
			height: 100,
			x: 0,
			y: 0,
			bottom: 100,
			right: 100,
			toJSON: () => ({})
		};
		const event = { clientX: 0, clientY: 0 } as MouseEvent;
		
		const result = getNormalizedMousePosition(event, rect);
		
		expect(result.x).toBeCloseTo(-1, 2);
		expect(result.y).toBeCloseTo(1, 2);
	});

	it('should normalize bottom-right corner to (1, -1)', () => {
		const rect: DOMRect = {
			left: 0,
			top: 0,
			width: 100,
			height: 100,
			x: 0,
			y: 0,
			bottom: 100,
			right: 100,
			toJSON: () => ({})
		};
		const event = { clientX: 100, clientY: 100 } as MouseEvent;
		
		const result = getNormalizedMousePosition(event, rect);
		
		expect(result.x).toBeCloseTo(1, 2);
		expect(result.y).toBeCloseTo(-1, 2);
	});
});

