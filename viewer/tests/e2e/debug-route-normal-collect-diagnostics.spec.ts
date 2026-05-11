import { expect, test } from '@playwright/test';

test.describe('debug route normal collect diagnostics', () => {
	test.afterEach(async ({ page }) => {
		await page.goto('about:blank').catch(() => undefined);
	});

	test('publishes normal UTCI results without claiming Python bin validity', async ({ page }) => {
		test.setTimeout(60_000);
		await page.goto('/debug?collect=normal', {
			waitUntil: 'domcontentloaded',
			timeout: 10_000
		});

		await expect(page.locator('[data-testid="on-demand-prototype-status"]')).toHaveCount(0);

		await expect
			.poll(
				async () =>
					page.evaluate(() => {
						const win = window as Window & {
							__normalUtciResults__?: {
								numPoints: number;
								numHours: number;
								monthIndex: number;
								utciByHour: number[][];
							};
							__onDemandPrototypeDiagnostics__?: Record<string, unknown>;
						};
						return {
							hasNormalResults: win.__normalUtciResults__ != null,
							numHours: win.__normalUtciResults__?.numHours ?? null,
							monthIndex: win.__normalUtciResults__?.monthIndex ?? null,
							hourCount: win.__normalUtciResults__?.utciByHour?.length ?? null,
							hasOnDemandDiagnostics: win.__onDemandPrototypeDiagnostics__ != null,
							binComparisonValid:
								win.__onDemandPrototypeDiagnostics__?.binComparisonValid ?? null,
							pythonBaselineStatus:
								win.__onDemandPrototypeDiagnostics__?.pythonBaselineStatus ?? null
						};
					}),
				{ timeout: 60_000 }
			)
			.toMatchObject({
				hasNormalResults: true,
				numHours: 24,
				monthIndex: 7,
				hourCount: 24,
				hasOnDemandDiagnostics: false,
				binComparisonValid: null,
				pythonBaselineStatus: null
			});
	});
});
