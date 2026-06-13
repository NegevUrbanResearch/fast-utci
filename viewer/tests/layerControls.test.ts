import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/svelte/svelte5';
import LayerControls from '$lib/components/ui/LayerControls.svelte';
import { setDiscoveredLayers, resetLayerVisibility } from '$lib/stores/layerStore';
import { viewerStore } from '$lib/stores/viewerStore';

describe('LayerControls', () => {
	beforeEach(() => {
		resetLayerVisibility();
		setDiscoveredLayers([]);
		viewerStore.set({
			currentHour: 0,
			currentMonth: 7,
			colorMode: 'normalized',
			metricType: 'utci',
			utciVisible: true,
			analysisId: null,
			loading: false,
			error: null,
			theme: 'dark'
		});
	});

	it('does not render ignored layers in the fallback discovered-layer list', () => {
		setDiscoveredLayers(['ignored']);

		const { queryByText } = render(LayerControls, { placement: 'sidebar' });

		expect(queryByText('Ignored')).not.toBeInTheDocument();
	});

	it('shows roads in the layer UI and keeps roads and train tracks visible by default', () => {
		setDiscoveredLayers(['road', 'train_track']);

		render(LayerControls, { placement: 'sidebar' });

		expect(screen.getByText('Roads')).toBeInTheDocument();
		expect(
			screen.getByRole('button', { name: 'Toggle Roads layer visibility' })
		).toHaveAttribute('aria-pressed', 'true');
		expect(
			screen.getByRole('button', { name: 'Toggle Train Tracks layer visibility' })
		).toHaveAttribute('aria-pressed', 'true');
	});
});
