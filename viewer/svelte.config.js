import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// Configure for GitHub Pages deployment
		adapter: adapter({
			// Default options for static adapter
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		paths: {
			// Set base path for GitHub Pages (repo name + /viewer/build/)
			base: process.env.NODE_ENV === 'production' ? '/fast-utci/viewer/build' : ''
		}
	}
};

export default config;
