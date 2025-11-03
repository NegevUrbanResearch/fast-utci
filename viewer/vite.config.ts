import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		fs: {
			// Allow serving files from the parent directory (where data/ folder is)
			allow: ['..']
		},
		// Disable HMR in test mode to avoid issues with vitest
		hmr: process.env.VITEST ? false : undefined
	}
});
