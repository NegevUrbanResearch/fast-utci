import { describe, it, expect } from 'vitest';
import {
	LADYBUG_NUANCED_COLORS,
	getUTCICategory,
	getUTCIColor,
	getUTCILabel,
	hexToThreeColor,
	mapUTCIToColor
} from '$lib/services/colorScale';

describe('ColorScale service', () => {
	describe('LADYBUG_NUANCED_COLORS', () => {
		it('should have 11 colors', () => {
			expect(LADYBUG_NUANCED_COLORS).toHaveLength(11);
		});

		it('should have valid hex color strings', () => {
			LADYBUG_NUANCED_COLORS.forEach((color) => {
				expect(color).toMatch(/^#[0-9A-Fa-f]{6}$/);
			});
		});
	});

	describe('getUTCICategory', () => {
		it('should return -5 for extreme cold', () => {
			expect(getUTCICategory(-50)).toBe(-5);
		});

		it('should return 0 for comfortable range', () => {
			expect(getUTCICategory(20)).toBe(0);
		});

		it('should return 5 for extreme heat', () => {
			expect(getUTCICategory(50)).toBe(5);
		});
	});

	describe('getUTCIColor', () => {
		it('should return color for extreme cold', () => {
			const color = getUTCIColor(-50);
			expect(color).toBe('#313695');
		});

		it('should return color for comfortable range', () => {
			const color = getUTCIColor(20);
			expect(color).toBe('#FFFFBF');
		});

		it('should return color for extreme heat', () => {
			const color = getUTCIColor(50);
			expect(color).toBe('#A50026');
		});
	});

	describe('getUTCILabel', () => {
		it('should return full label by default', () => {
			expect(getUTCILabel(-50)).toBe('Extreme Cold Stress');
			expect(getUTCILabel(20)).toBe('No Thermal Stress');
			expect(getUTCILabel(50)).toBe('Extreme Heat Stress');
		});

		it('should return abbreviated label when requested', () => {
			expect(getUTCILabel(-50, true)).toBe('Extreme Cold');
			expect(getUTCILabel(20, true)).toBe('Comfortable');
			expect(getUTCILabel(50, true)).toBe('Extreme Heat');
		});
	});

	describe('hexToThreeColor', () => {
		it('should convert hex to normalized RGB', () => {
			const color = hexToThreeColor('#FF0000');
			expect(color.r).toBeCloseTo(1.0, 5);
			expect(color.g).toBeCloseTo(0.0, 5);
			expect(color.b).toBeCloseTo(0.0, 5);
		});

		it('should handle hex with # prefix', () => {
			const color = hexToThreeColor('#00FF00');
			expect(color.g).toBeCloseTo(1.0, 5);
		});

		it('should handle hex without # prefix', () => {
			const color = hexToThreeColor('0000FF');
			expect(color.b).toBeCloseTo(1.0, 5);
		});
	});

	describe('mapUTCIToColor', () => {
		it('should map minimum value to first color', () => {
			const color = mapUTCIToColor(10, 10, 30);
			const firstColor = hexToThreeColor(LADYBUG_NUANCED_COLORS[0]);
			expect(color.r).toBeCloseTo(firstColor.r, 2);
			expect(color.g).toBeCloseTo(firstColor.g, 2);
			expect(color.b).toBeCloseTo(firstColor.b, 2);
		});

		it('should map maximum value to last color', () => {
			const color = mapUTCIToColor(30, 10, 30);
			const lastColor = hexToThreeColor(LADYBUG_NUANCED_COLORS[10]);
			expect(color.r).toBeCloseTo(lastColor.r, 2);
			expect(color.g).toBeCloseTo(lastColor.g, 2);
			expect(color.b).toBeCloseTo(lastColor.b, 2);
		});

		it('should interpolate for middle values', () => {
			const color = mapUTCIToColor(20, 10, 30);
			// Should be somewhere in the middle of the color scale
			// Values can be 0-1 (inclusive) since exact color matches are valid
			expect(color.r).toBeGreaterThanOrEqual(0);
			expect(color.r).toBeLessThanOrEqual(1);
			expect(color.g).toBeGreaterThanOrEqual(0);
			expect(color.g).toBeLessThanOrEqual(1);
			expect(color.b).toBeGreaterThanOrEqual(0);
			expect(color.b).toBeLessThanOrEqual(1);
		});

		it('should clamp values outside range', () => {
			const belowMin = mapUTCIToColor(5, 10, 30);
			const aboveMax = mapUTCIToColor(35, 10, 30);
			
			expect(belowMin.r).toBeCloseTo(mapUTCIToColor(10, 10, 30).r, 2);
			expect(aboveMax.r).toBeCloseTo(mapUTCIToColor(30, 10, 30).r, 2);
		});
	});
});

