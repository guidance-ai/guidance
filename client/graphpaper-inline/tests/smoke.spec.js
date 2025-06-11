// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Widget Smoke Test', () => {
  test('test harness loads and basic API works', async ({ page }) => {
    // Navigate to test harness using file:// protocol for now
    await page.goto(`file://${__dirname}/../src/test-utils/test-harness.html`);
    
    // Check page loads
    await expect(page.locator('h3')).toContainText('Widget Test Harness');
    
    // Wait for test API to be ready
    await page.waitForFunction(() => window.testAPIReady === true, { timeout: 5000 });
    
    // Check that testAPI exists
    const hasTestAPI = await page.evaluate(() => typeof window.testAPI === 'object');
    expect(hasTestAPI).toBe(true);
    
    // Inject simple test data - just one token
    const simpleMockData = [
      {
        "class_name": "TextOutput",
        "value": "Hello",
        "is_input": false,
        "is_generated": true,
        "prob": 0.9
      }
    ];
    
    // Inject the data
    const result = await page.evaluate((data) => {
      return window.testAPI.injectMockData('smoke-test', data);
    }, simpleMockData);
    
    // Check injection worked
    expect(result.success).toBe(true);
    expect(result.componentCount).toBe(1);
    
    // Wait for widget to render
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    // Check that we have a token grid
    const tokenGrid = page.locator('.token-grid');
    await expect(tokenGrid).toBeVisible();
    
    // Check that our token appears
    const token = page.locator('.token').filter({ hasText: 'Hello' });
    await expect(token).toBeVisible();
    
    // Check status updates
    await expect(page.locator('#current-scenario')).toContainText('smoke-test');
    await expect(page.locator('#component-count')).toContainText('1');
  });
});