import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
	testDir: 'tests/e2e',
	fullyParallel: false,
	workers: 1,
	timeout: 300_000,
	reporter: 'list',
	webServer: {
		command: 'npm run dev -- --port 4173',
		url: 'http://localhost:4173',
		reuseExistingServer: false,
		timeout: 120_000
	},
	use: {
		baseURL: 'http://localhost:4173',
		trace: 'retain-on-failure',
		navigationTimeout: 45_000
	},
	projects: [
		{
			name: 'chromium',
			use: {
				...devices['Desktop Chrome'],
				headless: false,
				launchOptions: {
					args: [
						'--enable-unsafe-webgpu'
					]
				}
			}
		}
	]
});
