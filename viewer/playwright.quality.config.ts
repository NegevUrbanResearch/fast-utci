import { defineConfig } from '@playwright/test';
import baseConfig from './playwright.config';

// Root owns the server session and its teardown. Tests never spawn an npm wrapper.
export default defineConfig({
	...baseConfig,
	testMatch: '**/*.spec.ts',
	webServer: undefined,
	use: {
		...baseConfig.use,
		baseURL: process.env.QUALITY_BASE_URL ?? 'http://localhost:5173'
	}
});
