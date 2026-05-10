<script lang="ts">
	import nurLogo from "$lib/assets/Nur Logo white.svg";
	import mitLogo from "$lib/assets/MIT.svg";
	import bguLogo from "$lib/assets/bgu-logo.svg";
	import sceLogo from "$lib/assets/sce-logo.svg";

	export let debugLabel: string | null = null;
	export let mainViewportElement: HTMLElement | null = null;
	export let showTimeSection = true;
	export let showSidebarExtraSection = true;
</script>

<div class="viewer-shell">
	<header class="app-header">
		<div class="header-left">
			<div class="partner-logos">
				<img
					src={nurLogo}
					alt="NUR Negev Urban Research"
					class="logo logo-nur"
				/>
				<img src={bguLogo} alt="BGU" class="logo logo-bgu" />
				<img src={mitLogo} alt="MIT" class="logo logo-mit" />
				<img src={sceLogo} alt="SCE" class="logo logo-sce" />
			</div>
		</div>
		<div class="header-center">
			<div
				class="header-title"
				class:header-title-with-debug={Boolean(debugLabel)}
			>
				<div class="logo-final">
					<div class="text">Score.CH</div>
					<div class="underline-grad"></div>
				</div>
				{#if debugLabel}
					<div class="debug-label">{debugLabel}</div>
				{/if}
			</div>
		</div>
		<div class="header-right">
			<slot name="headerRight" />
		</div>
	</header>

	<div class="app-body">
		<aside class="app-sidebar">
			{#if $$slots.scenario}
				<div class="sidebar-section">
					<slot name="scenario" />
				</div>
			{/if}

			{#if $$slots.analytics}
				<div class="sidebar-section analytics-section">
					<slot name="analytics" />
				</div>
			{/if}

			{#if $$slots.layers}
				<div class="sidebar-section">
					<slot name="layers" />
				</div>
			{/if}

			{#if $$slots.time && showTimeSection}
				<div class="sidebar-section">
					<slot name="time" />
				</div>
			{/if}

			{#if $$slots.sidebarExtra && showSidebarExtraSection}
				<div class="sidebar-section">
					<slot name="sidebarExtra" />
				</div>
			{/if}
		</aside>

		<main class="app-main" bind:this={mainViewportElement}>
			<slot />
			{#if $$slots.legend}
				<div class="legend-container">
					<slot name="legend" />
				</div>
			{/if}
			<slot name="tooltip" />
			<slot name="overlays" />
			<slot name="viewport" />
		</main>
	</div>
</div>

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		overflow: hidden;
		width: 100%;
		height: 100%;
		font-family: var(--font-family);
		background: var(--color-bg-page);
		color: var(--color-text-primary);
	}

	.viewer-shell {
		width: 100vw;
		height: 100vh;
		display: flex;
		flex-direction: column;
		background: radial-gradient(
				circle at top left,
				rgba(56, 189, 248, 0.18),
				transparent 55%
			),
			var(--color-bg-page);
	}

	.app-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 8px 18px;
		background: var(--color-bg-header);
		backdrop-filter: blur(16px);
		z-index: 10;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
		gap: 20px;
	}

	.header-left {
		display: flex;
		align-items: center;
		flex: 1 1 0;
		min-width: 0;
		justify-content: flex-start;
		overflow: hidden;
	}

	.header-center {
		display: flex;
		align-items: center;
		flex: 0 0 auto;
		justify-content: center;
		min-width: 0;
	}

	.header-right {
		display: flex;
		align-items: center;
		flex: 1 1 0;
		min-width: 0;
		justify-content: flex-end;
	}

	.header-title {
		display: flex;
		align-items: center;
		padding: 0;
		background: transparent;
		border: none;
		box-shadow: none;
	}

	.header-title-with-debug {
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
	}

	.logo-final {
		position: relative;
		font-family: "Space Grotesk", sans-serif;
	}

	.logo-final .text {
		font-size: 34px;
		font-weight: 700;
		color: var(--color-text-primary);
		letter-spacing: -0.03em;
		padding-bottom: 2px;
	}

	.logo-final .underline-grad {
		position: absolute;
		bottom: -2px;
		left: 0;
		right: 0;
		height: 4px;
		background: linear-gradient(
			90deg,
			#313695,
			#4575b4,
			#74add1,
			#abd9e9,
			#e0f3f8,
			#ffffbf,
			#fee090,
			#fdae61,
			#f46d43,
			#d73027,
			#a50026
		);
		border-radius: 2px;
		opacity: 0.9;
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
	}

	.debug-label {
		font-size: 11px;
		color: var(--color-text-secondary);
		margin-top: 6px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.partner-logos {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: nowrap;
		max-width: 100%;
	}

	.logo {
		height: 30px;
		object-fit: contain;
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.4));
		display: block;
	}

	.logo-nur {
		height: 50px;
	}

	.logo-bgu {
		height: 35px;
	}

	.logo-sce {
		height: 30px;
		filter: invert(1) drop-shadow(0 0 4px rgba(0, 0, 0, 0.6));
	}

	.app-body {
		flex: 1;
		display: grid;
		grid-template-columns: minmax(320px, 320px) 1fr;
		grid-template-areas: "sidebar main";
		height: 100%;
		overflow: hidden;
		position: relative;
	}

	.app-sidebar {
		grid-area: sidebar;
		background: var(--color-bg-sidebar);
		padding: 12px 10px;
		display: flex;
		flex-direction: column;
		gap: 10px;
		overflow-y: auto;
		overflow-x: hidden;
		scrollbar-gutter: stable;
		width: 320px;
		min-width: 320px;
		max-width: 320px;
		box-sizing: border-box;
		flex-shrink: 0;
		contain: layout size;
		position: relative;
		box-shadow: 2px 0 12px rgba(0, 0, 0, 0.12);
	}

	.app-main {
		grid-area: main;
		position: relative;
		background: var(--color-bg-page);
		min-width: 0;
		overflow: hidden;
	}

	:global(.viewer-shell .sidebar-section) {
		background: var(--color-bg-panel);
		border-radius: var(--radius-panel);
		box-shadow: var(--shadow-panel);
		padding: 10px 12px;
		min-width: 0;
		max-width: 100%;
		box-sizing: border-box;
	}

	:global(.viewer-shell .section-header) {
		font-size: var(--font-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		margin-bottom: 8px;
		color: var(--color-text-secondary);
	}

	:global(.viewer-shell .section-subtitle) {
		font-size: var(--font-sm);
		color: var(--color-text-muted);
		margin-bottom: 8px;
	}

	:global(.viewer-shell .section-subtitle.error) {
		color: var(--color-danger);
	}

	:global(.viewer-shell .section-header-toggle) {
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 6px;
		background: transparent;
		border: none;
		padding: 0;
		cursor: pointer;
		color: var(--color-text-secondary);
		font-family: var(--font-family);
	}

	:global(.viewer-shell .analytics-section) {
		padding-top: 8px;
	}

	:global(.viewer-shell .analytics-section .section-header) {
		margin-bottom: 4px;
	}

	:global(.viewer-shell .chevron) {
		transition: transform 0.15s ease;
	}

	:global(.viewer-shell .chevron.open) {
		transform: rotate(180deg);
	}

	.legend-container {
		position: absolute;
		bottom: 20px;
		right: 20px;
		z-index: var(--z-tooltip);
	}

	:global(.viewer-shell .overlay-message) {
		position: absolute;
		top: 16px;
		left: 50%;
		transform: translateX(-50%);
		z-index: var(--z-tooltip);
		padding: 10px 16px;
		border-radius: 999px;
		background: var(--color-bg-panel);
		color: var(--color-text-primary);
		box-shadow: var(--shadow-panel);
		font-size: 13px;
	}

	:global(.viewer-shell .overlay-message.error) {
		border: 1px solid var(--color-danger);
	}

	:global(.viewer-shell .model-loading-backdrop) {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		z-index: calc(var(--z-tooltip) - 1);
		background: rgba(17, 24, 39, 0.4);
		backdrop-filter: blur(12px);
		-webkit-backdrop-filter: blur(12px);
		pointer-events: none;
		transition: opacity 0.25s ease-out;
	}

	:global(.viewer-shell .model-loading-overlay) {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		z-index: var(--z-tooltip);
		min-width: 180px;
		padding: 14px 18px;
		border-radius: 14px;
		background: rgba(17, 24, 39, 0.82);
		backdrop-filter: blur(10px);
		color: white;
		box-shadow:
			0 14px 30px rgba(0, 0, 0, 0.35),
			0 0 0 1px rgba(255, 255, 255, 0.05);
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 10px;
		text-align: center;
	}

	:global(.viewer-shell .model-loading-overlay .loading-text) {
		font-size: 13px;
		letter-spacing: 0.04em;
	}

	:global(.viewer-shell .spinner) {
		width: 36px;
		height: 36px;
		border-radius: 50%;
		border: 3px solid rgba(255, 255, 255, 0.18);
		border-top-color: var(--color-accent);
		animation: spin 0.9s linear infinite;
	}

	@keyframes spin {
		from {
			transform: rotate(0deg);
		}
		to {
			transform: rotate(360deg);
		}
	}

	@media (max-width: 1400px) {
		.app-header {
			padding: 8px 14px;
			gap: 14px;
		}

		.partner-logos {
			gap: 8px;
		}

		.logo {
			height: 30px;
		}

		.logo-nur {
			height: 35px;
		}

		.logo-bgu {
			height: 39px;
		}

		.logo-sce {
			height: 32px;
		}

		.logo-final .text {
			font-size: 28px;
		}
	}

	@media (max-width: 1100px) {
		.app-header {
			gap: 10px;
		}

		.partner-logos {
			gap: 6px;
		}

		.logo-mit,
		.logo-sce {
			display: none;
		}

		.header-title {
			padding: 3px 10px;
		}

		.logo-final .text {
			font-size: 24px;
			letter-spacing: -0.02em;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		:global(.viewer-shell .model-loading-backdrop) {
			transition: none;
		}
	}

	:global(html[data-theme="dark"] .viewer-shell .logo-nur) {
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.6));
	}

	:global(html[data-theme="light"] .viewer-shell .logo-nur) {
		filter: brightness(0) drop-shadow(0 0 4px rgba(0, 0, 0, 0.3));
	}

	:global(html[data-theme="dark"] .viewer-shell .logo-bgu) {
		filter: drop-shadow(0 0 3px rgba(0, 0, 0, 0.55));
	}

	:global(html[data-theme="light"] .viewer-shell .logo-bgu) {
		filter: drop-shadow(0 0 3px rgba(15, 23, 42, 0.45));
	}

	:global(html[data-theme="dark"] .viewer-shell .logo-mit) {
		filter: invert(1) drop-shadow(0 0 4px rgba(0, 0, 0, 0.6));
	}

	:global(html[data-theme="light"] .viewer-shell .logo-mit) {
		filter: drop-shadow(0 0 4px rgba(0, 0, 0, 0.4));
	}

	:global(html[data-theme="dark"] .viewer-shell .app-sidebar .sidebar-section) {
		box-shadow:
			0 14px 30px rgba(15, 23, 42, 0.7),
			0 0 0 1px rgba(248, 250, 252, 0.03);
	}
</style>
