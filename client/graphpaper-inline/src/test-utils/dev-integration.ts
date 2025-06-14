// Development integration for using mock generator in the main app
import { 
  mockGenerator, 
  createBacktrackTest, 
  createRoleConfusionTest, 
  createMultimodalTest 
} from './mock-generator';

// Predefined test scenarios that can be easily swapped in App.svelte
export const testScenarios = {
  // Original static mock (for backwards compatibility)
  original: () => {
    // This would import from the original mocks.ts if needed
    return [];
  },

  // Simple conversation for basic testing
  simple: () => mockGenerator()
    .addRole({
      name: "user",
      tokens: [{ text: "Hello, how are you?" }]
    })
    .addRole({
      name: "assistant", 
      tokens: [
        { text: "I'm", prob: 0.9, is_generated: true },
        { text: " doing", prob: 0.8, is_generated: true },
        { text: " well", prob: 0.9, is_generated: true },
        { text: ",", prob: 0.7, is_generated: true },
        { text: " thank", prob: 0.85, is_generated: true },
        { text: " you", prob: 0.9, is_generated: true },
        { text: "!", prob: 0.6, is_generated: true }
      ]
    })
    .build(),

  // Test the backtracking fix specifically  
  backtrackFix: () => createBacktrackTest().build(),

  // Test role confusion after backtracking
  roleConfusion: () => createRoleConfusionTest().build(),

  // Long conversation for performance testing
  longConversation: () => {
    let generator = mockGenerator();
    for (let i = 0; i < 10; i++) {
      generator = generator
        .addRole({
          name: "user",
          tokens: [{ text: `User message ${i + 1}` }]
        })
        .addRole({
          name: "assistant",
          tokens: [
            { text: `Assistant`, prob: 0.9, is_generated: true },
            { text: ` response`, prob: 0.8, is_generated: true },
            { text: ` ${i + 1}`, prob: 0.7, is_generated: true }
          ]
        });
    }
    return generator.build();
  },

  // Multiple backtracks scenario
  multipleBacktracks: () => mockGenerator()
    .addRole({
      name: "user",
      tokens: [{ text: "Count to 10" }]
    })
    .addRole({
      name: "assistant",
      tokens: [
        { text: "1", prob: 0.9, is_generated: true },
        { text: " 2", prob: 0.9, is_generated: true },
        { text: " 3", prob: 0.9, is_generated: true },
        { text: " 5", prob: 0.2, is_generated: true } // Wrong - should be 4
      ]
    })
    .addBacktrack({ n_tokens: 1, at_position: -1 })
    .addTokens([{ text: " 4", prob: 0.95, is_generated: true }])
    .addTokens([
      { text: " 5", prob: 0.9, is_generated: true },
      { text: " 6", prob: 0.9, is_generated: true },
      { text: " 8", prob: 0.1, is_generated: true } // Wrong again - should be 7
    ])
    .addBacktrack({ n_tokens: 1, at_position: -1 })
    .addTokens([{ text: " 7", prob: 0.95, is_generated: true }])
    .build(),

  // Multimodal content
  multimodal: () => createMultimodalTest().build(),

  // Varying probabilities for visual testing
  probabilities: () => mockGenerator()
    .addRole({
      name: "assistant",
      tokens: [
        { text: "Very", prob: 0.95 },      // High confidence - should be dark
        { text: " high", prob: 0.9 },      
        { text: " confidence", prob: 0.85 },
        { text: " medium", prob: 0.5 },    // Medium confidence
        { text: " confidence", prob: 0.3 }, // Low confidence  
        { text: " very", prob: 0.1 },      // Very low - should be light
        { text: " low", prob: 0.05 }       // Extremely low
      ].map(token => ({ ...token, is_generated: true }))
    })
    .build(),

  // Edge cases
  edgeCases: () => mockGenerator()
    .addTokens([
      { text: "", prob: 0.5 }, // Empty token
      { text: "\n", prob: 0.8 }, // Newline
      { text: "   ", prob: 0.3 }, // Spaces only
      { text: "🎉", prob: 0.7 }, // Emoji
      { text: "Very long token that might cause layout issues in the UI", prob: 0.4 }
    ])
    .build()
};

// Helper to get scenario by name with error handling
export function getTestScenario(name: keyof typeof testScenarios) {
  if (!(name in testScenarios)) {
    console.warn(`Test scenario '${name}' not found, using 'simple'`);
    return testScenarios.simple();
  }
  return testScenarios[name]();
}

// For use in App.svelte, just change this line to switch scenarios:
// Example: appState.components = getTestScenario('backtrackFix');
export const currentTestScenario = 'simple'; // Change this to test different scenarios