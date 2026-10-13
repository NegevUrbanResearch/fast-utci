import { defineConfig } from 'vitest/config';
import { defaultClientConditions } from 'vite';
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
		include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
		globals: true,
		environment: 'jsdom',
		setupFiles: ['./tests/setup.ts']
	},
	resolve: {
		conditions: defaultClientConditions,
		alias: {
			$lib: resolve('./src/lib'),
			'$app/paths': resolve('./tests/mocks/app-paths.ts')
		}
	}
});


