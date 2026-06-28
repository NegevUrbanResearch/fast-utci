import { describe, expect, it } from 'vitest';
import { render } from '@testing-library/svelte/svelte5';
import ScenarioSelector from '$lib/components/ui/ScenarioSelector.svelte';

describe('ScenarioSelector', () => {
	it('renders scenario controls when categories are available', () => {
		const { getByText } = render(ScenarioSelector, {
			projectId: 'Ben-Gurion',
			categories: [
				{
					value: 'existing_buildings',
					label: 'Existing buildings with added mass',
					description: 'Current buildings made higher'
				}
			]
		});

		expect(getByText('Browse variants')).toBeInTheDocument();
	});

	it('renders nothing when no scenario categories are available', () => {
		const { queryByText } = render(ScenarioSelector, {
			projectId: 'Ness-Tziona',
			categories: []
		});

		expect(queryByText('Browse variants')).not.toBeInTheDocument();
		expect(queryByText('No scenario selected')).not.toBeInTheDocument();
	});
});
