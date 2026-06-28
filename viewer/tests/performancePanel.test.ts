import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render } from '@testing-library/svelte/svelte5';
import PerformancePanel from '$lib/components/ui/PerformancePanel.svelte';
import { EMPTY_PERFORMANCE_SNAPSHOT, performanceStore } from '$lib/stores/performanceStore';

describe('PerformancePanel', () => {
	beforeEach(() => {
		performanceStore.set({
			...EMPTY_PERFORMANCE_SNAPSHOT,
			status: 'ready',
			gridSizeMeters: 2,
			pointCount: 1234
		});
	});

	it('exposes a compact discrete live-grid resolution slider without duplicate status or resolution text', async () => {
		const onGridResolutionChange = vi.fn();
		const { container, getByTestId, getByText, queryByText } = render(PerformancePanel as never, {
			selectedGridResolutionMeters: 2,
			onGridResolutionChange
		});

		const slider = getByTestId('performance-grid-resolution-slider') as HTMLInputElement;

		expect(slider.getAttribute('aria-label')).toBe('Live UTCI grid resolution');
		expect(slider.value).toBe('4');
		expect(getByText('2 m')).toBeTruthy();
		expect(queryByText('2 m requested')).toBeNull();
		expect(queryByText('Ready')).toBeNull();
		expect(queryByText('Preparing')).toBeNull();
		expect(queryByText('Needs attention')).toBeNull();
		expect(getByText('Grid points')).toBeTruthy();
		expect(getByText('1,234 pts')).toBeTruthy();
		expect(queryByText('Grid size')).toBeNull();
		expect(queryByText('2 m (1,234 pts)')).toBeNull();
		expect(container.textContent?.match(/2 m/g)?.length).toBe(1);

		await fireEvent.input(slider, { target: { value: '6' } });

		expect(onGridResolutionChange).toHaveBeenCalledWith(0.5);
	});
});
