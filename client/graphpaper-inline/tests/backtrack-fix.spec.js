// @ts-check
const { test, expect } = require('@playwright/test');

// Import mock generator logic (we'll include it inline for now)
const mockGenerator = {
  createBacktrackScenario() {
    return [
      // User role
      {
        "class_name": "RoleOpenerInput",
        "name": "user",
        "text": "<|user|>\n",
        "closer_text": "<|end|>\n"
      },
      {
        "class_name": "TextOutput",
        "value": "<|user|>\n",
        "is_input": true,
        "is_generated": false,
        "is_force_forwarded": false
      },
      {
        "class_name": "TextOutput",
        "value": "What is 2+2?",
        "is_input": true,
        "is_generated": false,
        "is_force_forwarded": false
      },
      {
        "class_name": "RoleCloserInput",
        "name": "user",
        "text": "<|end|>\n"
      },
      {
        "class_name": "TextOutput",
        "value": "<|end|>\n",
        "is_input": true,
        "is_generated": false,
        "is_force_forwarded": false
      },
      
      // Assistant role with wrong answer
      {
        "class_name": "RoleOpenerInput",
        "name": "assistant",
        "text": "<|assistant|>\n",
        "closer_text": "<|end|>\n"
      },
      {
        "class_name": "TextOutput",
        "value": "<|assistant|>\n",
        "is_input": true,
        "is_generated": false,
        "is_force_forwarded": false
      },
      {
        "class_name": "TextOutput",
        "value": "2",
        "is_input": false,
        "is_generated": true,
        "prob": 0.9
      },
      {
        "class_name": "TextOutput",
        "value": "+",
        "is_input": false,
        "is_generated": true,
        "prob": 0.8
      },
      {
        "class_name": "TextOutput",
        "value": "2",
        "is_input": false,
        "is_generated": true,
        "prob": 0.9
      },
      {
        "class_name": "TextOutput",
        "value": " equals",
        "is_input": false,
        "is_generated": true,
        "prob": 0.7
      },
      {
        "class_name": "TextOutput",
        "value": " 5",
        "is_input": false,
        "is_generated": true,
        "prob": 0.1  // Low probability - wrong answer
      },
      
      // Backtrack the wrong answer
      {
        "class_name": "Backtrack",
        "n_tokens": 1,
        "bytes": "ZmFrZS1kYXRh" // fake base64
      },
      
      // Correct answer after backtrack
      {
        "class_name": "TextOutput",
        "value": " 4",
        "is_input": false,
        "is_generated": true,
        "prob": 0.95  // High probability - correct answer
      }
    ];
  },
  
  createRoleConfusionScenario() {
    return [
      // User role
      {
        "class_name": "RoleOpenerInput",
        "name": "user",
        "text": "<|user|>\n",
        "closer_text": "<|end|>\n"
      },
      {
        "class_name": "TextOutput",
        "value": "Hello",
        "is_input": true,
        "is_generated": false
      },
      
      // Backtrack removing 2 tokens
      {
        "class_name": "Backtrack",
        "n_tokens": 2,
        "bytes": "ZmFrZS1kYXRh"
      },
      
      // Assistant role after backtrack
      {
        "class_name": "RoleOpenerInput",
        "name": "assistant",
        "text": "<|assistant|>\n",
        "closer_text": "<|end|>\n"
      },
      {
        "class_name": "TextOutput",
        "value": "Hi there!",
        "is_input": false,
        "is_generated": true,
        "prob": 0.9
      }
    ];
  }
};

test.describe('Widget Backtracking Fix', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to test harness using file:// protocol
    await page.goto(`file://${__dirname}/../src/test-utils/test-harness.html`);
    
    // Wait for test API to be ready
    await page.waitForFunction(() => window.testAPIReady === true);
  });

  test('handles basic backtrack scenario correctly', async ({ page }) => {
    // Inject backtrack test data
    const mockData = mockGenerator.createBacktrackScenario();
    await page.evaluate((data) => {
      return window.testAPI.injectMockData('backtrack-basic', data);
    }, mockData);
    
    // Wait for widget to render
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    // Check that the widget rendered
    const tokenGrid = page.locator('.token-grid');
    await expect(tokenGrid).toBeVisible();
    
    // Check that we have both user and assistant roles
    const userRole = page.locator('.role-indicator[data-role="user"]');
    const assistantRole = page.locator('.role-indicator[data-role="assistant"]');
    await expect(userRole).toBeVisible();
    await expect(assistantRole).toBeVisible();
    
    // Check that backtrack indicator is present
    const backtrackIndicator = page.locator('.backtrack-indicator');
    await expect(backtrackIndicator).toBeVisible();
    await expect(backtrackIndicator).toContainText('BACKTRACK (1 tokens)');
    
    // Check that the correct answer "4" is present (not the wrong "5")
    const correctAnswer = page.locator('.token[data-role="assistant"]').filter({ hasText: ' 4' });
    const wrongAnswer = page.locator('.token[data-role="assistant"]').filter({ hasText: ' 5' });
    
    await expect(correctAnswer).toBeVisible();
    await expect(wrongAnswer).not.toBeVisible(); // Should be removed by backtrack
  });

  test('assigns tokens to correct roles after backtrack', async ({ page }) => {
    // Use the role confusion scenario
    const mockData = mockGenerator.createRoleConfusionScenario();
    await page.evaluate((data) => {
      return window.testAPI.injectMockData('role-confusion', data);
    }, mockData);
    
    // Wait for widget to render
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    // Get widget state for analysis
    const widgetState = await page.evaluate(() => window.testAPI.getWidgetState());
    
    // Check that we have the expected roles
    expect(widgetState.roles).toHaveLength(2); // user and assistant
    expect(widgetState.roles[0].role).toBe('user');
    expect(widgetState.roles[1].role).toBe('assistant');
    
    // Check that tokens are assigned to correct roles
    const assistantTokens = widgetState.tokens.filter(token => token.role === 'assistant');
    expect(assistantTokens).toHaveLength(1);
    expect(assistantTokens[0].text).toBe('Hi there!');
    expect(assistantTokens[0].isGenerated).toBe(true);
    
    // Check backtrack indicator
    expect(widgetState.backtracks).toHaveLength(1);
    expect(widgetState.backtracks[0].tokens).toBe(2);
  });

  test('visual appearance of tokens with different probabilities', async ({ page }) => {
    const mockData = mockGenerator.createBacktrackScenario();
    await page.evaluate((data) => {
      return window.testAPI.injectMockData('probability-visual', data);
    }, mockData);
    
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    // Check that high probability tokens are more opaque
    const highProbToken = page.locator('.token').filter({ hasText: ' 4' }); // prob 0.95
    const lowProbToken = page.locator('.token').filter({ hasText: ' equals' }); // prob 0.7
    
    // Get computed styles
    const highProbOpacity = await highProbToken.evaluate(el => getComputedStyle(el).opacity);
    const lowProbOpacity = await lowProbToken.evaluate(el => getComputedStyle(el).opacity);
    
    // High probability should have higher opacity
    expect(parseFloat(highProbOpacity)).toBeGreaterThan(parseFloat(lowProbOpacity));
  });

  test('tokens are clickable and provide debug info', async ({ page }) => {
    const mockData = mockGenerator.createBacktrackScenario();
    await page.evaluate((data) => {
      return window.testAPI.injectMockData('clickable-tokens', data);
    }, mockData);
    
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    // Set up console listener to capture click events
    const consoleMessages = [];
    page.on('console', msg => {
      if (msg.text().includes('Token clicked:')) {
        consoleMessages.push(msg.text());
      }
    });
    
    // Click on a generated token
    const tokenToClick = page.locator('.token[data-is-generated="true"]').first();
    await tokenToClick.click();
    
    // Wait a bit for console message
    await page.waitForTimeout(100);
    
    // Check that click was logged
    expect(consoleMessages.length).toBeGreaterThan(0);
    expect(consoleMessages[0]).toContain('Token clicked:');
  });

  test('handles empty components gracefully', async ({ page }) => {
    await page.evaluate(() => {
      return window.testAPI.injectMockData('empty', []);
    });
    
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    // Should show empty state without errors
    const tokenGrid = page.locator('.token-grid');
    await expect(tokenGrid).toBeVisible();
    
    const tokens = page.locator('.token');
    await expect(tokens).toHaveCount(0);
  });

  test('performance with large token sequences', async ({ page }) => {
    // Generate a large sequence of tokens
    const largeMockData = [];
    
    // Add assistant role
    largeMockData.push({
      "class_name": "RoleOpenerInput",
      "name": "assistant",
      "text": "<|assistant|>\n",
      "closer_text": "<|end|>\n"
    });
    
    // Add 100 tokens
    for (let i = 0; i < 100; i++) {
      largeMockData.push({
        "class_name": "TextOutput",
        "value": ` token${i}`,
        "is_input": false,
        "is_generated": true,
        "prob": Math.random()
      });
    }
    
    const startTime = Date.now();
    
    await page.evaluate((data) => {
      return window.testAPI.injectMockData('large-sequence', data);
    }, largeMockData);
    
    await page.evaluate(() => window.testAPI.waitForWidget());
    
    const renderTime = Date.now() - startTime;
    
    // Should render within reasonable time (2 seconds)
    expect(renderTime).toBeLessThan(2000);
    
    // Should have all tokens
    const tokens = page.locator('.token');
    await expect(tokens).toHaveCount(100);
  });
});