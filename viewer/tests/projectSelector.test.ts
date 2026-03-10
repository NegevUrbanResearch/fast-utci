import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/svelte/svelte5';
import ProjectSelector from '$lib/components/ui/ProjectSelector.svelte';

describe('ProjectSelector', () => {
	it('calls onSelect when project changes', async () => {
		const onSelect = vi.fn();
		const { getByTestId } = render(ProjectSelector, {
			analysisId: 'Ben-Gurion/20250815_grid_2m_fullday',
			onSelect
		});

		const projectSelect = getByTestId('project-select') as HTMLSelectElement;
		await fireEvent.change(projectSelect, { target: { value: 'Ness-Tziona' } });

		expect(onSelect).toHaveBeenCalledWith(
			'Ness-Tziona/exploded/nes_tziona_unblock_2'
		);
	});
});
