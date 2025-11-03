import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

export default defineConfig({
	plugins: [
		svelte({
			hot: false,
			emitCss: false,
			compilerOptions: {
				dev: false,
				hmr: false  // This disables HMR in the compiled output
			}
		})
	],
	test: {
		globals: true,
		environment: 'jsdom',
		setupFiles: ['./tests/setup.ts']
	},
	resolve: {
		alias: {
			$lib: resolve('./src/lib')
		}
	}
});


