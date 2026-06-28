<script lang="ts">
	import { viewerStore, setTheme } from '$lib/stores/viewerStore';

	function toggleTheme() {
		const current = $viewerStore.theme ?? 'dark';
		const next: 'dark' | 'light' = current === 'dark' ? 'light' : 'dark';
		setTheme(next);

		if (typeof window !== 'undefined') {
			window.localStorage.setItem('fast_utci_theme', next);
		}
	}
</script>

<button class="theme-toggle" type="button" on:click={toggleTheme} aria-label="Toggle color theme">
	<span class="label">{$viewerStore.theme === 'dark' ? 'Dark' : 'Light'}</span>
	<span class="pill" aria-hidden="true">
		<span class:knob-dark={$viewerStore.theme === 'dark'} class="knob"></span>
	</span>
</button>

<style>
	.theme-toggle {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 8px;
		border-radius: 999px;
		background: var(--color-bg-panel-soft);
		color: var(--color-text-secondary);
		font-size: 11px;
		font-family: var(--font-family);
		cursor: pointer;
		transition: background 0.15s ease;
	}

	.theme-toggle:hover {
		background: var(--color-accent-soft);
	}

	.label {
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.pill {
		width: 32px;
		height: 16px;
		border-radius: 999px;
		background: rgba(148, 163, 184, 0.35);
		position: relative;
	}

	.knob {
		position: absolute;
		top: 2px;
		left: 2px;
		width: 12px;
		height: 12px;
		border-radius: 999px;
		background: #ffffff;
		box-shadow: 0 1px 2px rgba(15, 23, 42, 0.5);
		transition: transform 0.18s ease;
	}

	.knob.knob-dark {
		transform: translateX(14px);
	}
</style>


