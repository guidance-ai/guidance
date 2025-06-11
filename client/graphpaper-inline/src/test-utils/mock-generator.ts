// Dynamic mock data generator for testing
import type { 
  NodeAttr, 
  TextOutput, 
  TokenOutput,
  RoleOpenerInput, 
  RoleCloserInput,
  AudioOutput,
  VideoOutput,
  ImageOutput,
  BacktrackMessage
} from '../stitch';

export interface MockToken {
  text: string;
  prob?: number;
  is_input?: boolean;
  is_generated?: boolean;
  is_force_forwarded?: boolean;
  latency_ms?: number;
}

export interface MockRole {
  name: string;
  opener_text?: string;
  closer_text?: string;
  tokens: MockToken[];
}

export interface BacktrackConfig {
  n_tokens: number;
  at_position: number; // Insert backtrack after this many components
}

export class MockDataGenerator {
  private components: NodeAttr[] = [];
  
  constructor() {
    this.reset();
  }

  reset(): MockDataGenerator {
    this.components = [];
    return this;
  }

  // Add a role-based conversation
  addRole(config: MockRole): MockDataGenerator {
    // Add role opener
    this.components.push({
      class_name: "RoleOpenerInput",
      name: config.name,
      text: config.opener_text || `<|${config.name}|>\n`,
      closer_text: config.closer_text || "<|end|>\n"
    } as RoleOpenerInput);

    // Add opener text token
    if (config.opener_text || config.name) {
      this.components.push(this.createTextOutput({
        text: config.opener_text || `<|${config.name}|>\n`,
        is_input: true,
        is_generated: false,
        is_force_forwarded: false,
        prob: 1.0
      }));
    }

    // Add content tokens
    config.tokens.forEach(token => {
      this.components.push(this.createTextOutput(token));
    });

    // Add role closer
    this.components.push({
      class_name: "RoleCloserInput", 
      name: config.name,
      text: config.closer_text || "<|end|>\n"
    } as RoleCloserInput);

    // Add closer text token
    this.components.push(this.createTextOutput({
      text: config.closer_text || "<|end|>\n",
      is_input: true,
      is_generated: false,
      is_force_forwarded: false,
      prob: 1.0
    }));

    return this;
  }

  // Add raw tokens without roles
  addTokens(tokens: MockToken[]): MockDataGenerator {
    tokens.forEach(token => {
      this.components.push(this.createTextOutput(token));
    });
    return this;
  }

  // Add media content
  addImage(format: string = "png", base64Data?: string): MockDataGenerator {
    this.components.push({
      class_name: "ImageOutput",
      value: base64Data || this.generateFakeBase64(),
      format,
      is_input: false
    } as ImageOutput);
    return this;
  }

  addAudio(format: string = "wav", base64Data?: string): MockDataGenerator {
    this.components.push({
      class_name: "AudioOutput", 
      value: base64Data || this.generateFakeBase64(),
      format,
      is_input: false
    } as AudioOutput);
    return this;
  }

  addVideo(format: string = "mp4", base64Data?: string): MockDataGenerator {
    this.components.push({
      class_name: "VideoOutput",
      value: base64Data || this.generateFakeBase64(),
      format,
      is_input: false
    } as VideoOutput);
    return this;
  }

  // Add backtracking
  addBacktrack(config: BacktrackConfig): MockDataGenerator {
    // Insert backtrack at specified position
    const backtrack: BacktrackMessage = {
      class_name: "Backtrack",
      n_tokens: config.n_tokens,
      bytes: this.generateFakeBase64()
    };

    if (config.at_position >= this.components.length) {
      this.components.push(backtrack);
    } else {
      this.components.splice(config.at_position, 0, backtrack);
    }
    return this;
  }

  // Helper to create a conversation scenario
  createConversation(turns: Array<{role: string, message: string}>): MockDataGenerator {
    turns.forEach(turn => {
      this.addRole({
        name: turn.role,
        tokens: this.tokenizeMessage(turn.message)
      });
    });
    return this;
  }

  // Create a backtracking scenario for testing
  createBacktrackScenario(): MockDataGenerator {
    return this
      .addRole({
        name: "user",
        tokens: [{ text: "What is 2+2?" }]
      })
      .addRole({
        name: "assistant", 
        tokens: [
          { text: "2", prob: 0.9, is_generated: true },
          { text: "+", prob: 0.8, is_generated: true },
          { text: "2", prob: 0.9, is_generated: true },
          { text: " equals", prob: 0.7, is_generated: true },
          { text: " 5", prob: 0.1, is_generated: true } // Wrong answer to trigger backtrack
        ]
      })
      .addBacktrack({ n_tokens: 1, at_position: this.components.length })
      .addTokens([
        { text: " 4", prob: 0.9, is_generated: true } // Correct answer after backtrack
      ]);
  }

  // Create a role confusion scenario
  createRoleConfusionScenario(): MockDataGenerator {
    return this
      .addRole({
        name: "user",
        tokens: [{ text: "Hello" }]
      })
      .addBacktrack({ n_tokens: 2, at_position: this.components.length })
      .addRole({
        name: "assistant", 
        tokens: [{ text: "Hi there!" }]
      });
  }

  // Create multimodal scenario
  createMultimodalScenario(): MockDataGenerator {
    return this
      .addRole({
        name: "user",
        tokens: [{ text: "Describe this image:" }]
      })
      .addImage()
      .addRole({
        name: "assistant",
        tokens: [
          { text: "I can see", prob: 0.9, is_generated: true },
          { text: " an", prob: 0.8, is_generated: true },
          { text: " image", prob: 0.9, is_generated: true }
        ]
      })
      .addAudio()
      .addVideo();
  }

  // Get the generated components
  build(): NodeAttr[] {
    return [...this.components];
  }

  // Private helpers
  private createTextOutput(token: MockToken): TextOutput {
    return {
      class_name: "TextOutput",
      value: token.text,
      is_input: token.is_input ?? false,
      is_generated: token.is_generated ?? false, 
      is_force_forwarded: token.is_force_forwarded ?? false,
      latency_ms: token.latency_ms ?? 0.0
    } as TextOutput;
  }

  private createTokenOutput(token: MockToken): TokenOutput {
    return {
      class_name: "TokenOutput",
      value: token.text,
      is_input: token.is_input ?? false,
      is_generated: token.is_generated ?? false,
      is_force_forwarded: token.is_force_forwarded ?? false,
      latency_ms: token.latency_ms ?? 0.0,
      token: {
        token: token.text,
        bytes: this.generateFakeBase64(),
        prob: token.prob ?? 0.5,
        masked: false
      },
      top_k: [] // Could generate top_k alternatives if needed
    } as TokenOutput;
  }

  private tokenizeMessage(message: string): MockToken[] {
    // Simple word-based tokenization for mock data
    return message.split(' ').map((word, i) => ({
      text: i === 0 ? word : ` ${word}`,
      prob: Math.random() * 0.5 + 0.5, // Random prob between 0.5-1.0
      is_generated: true
    }));
  }

  private generateFakeBase64(): string {
    // Generate a short fake base64 string for testing
    return btoa("fake-data-" + Math.random().toString(36).substr(2, 9));
  }
}

// Convenience factory functions
export const mockGenerator = () => new MockDataGenerator();

export const createSimpleConversation = () => 
  mockGenerator().createConversation([
    { role: "user", message: "Hello world" },
    { role: "assistant", message: "Hi there! How can I help?" }
  ]);

export const createBacktrackTest = () => 
  mockGenerator().createBacktrackScenario();

export const createRoleConfusionTest = () =>
  mockGenerator().createRoleConfusionScenario();

export const createMultimodalTest = () =>
  mockGenerator().createMultimodalScenario();