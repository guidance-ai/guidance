// Mocks for interactive testing

import {type TextOutput, type RoleOpenerInput, type RoleCloserInput } from './stitch';

export const mockNodeAttrs: Array<RoleCloserInput | RoleOpenerInput | TextOutput> = [
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
    "is_force_forwarded": false,
    "latency_ms": 0,
  },
  {
    "class_name": "TextOutput",
    "value": "What is the capital of France?",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "latency_ms": 0,
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
    "is_force_forwarded": false,
    "latency_ms": 0,
  },
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
    "is_force_forwarded": false,
    "latency_ms": 0,
  },
  {
    "class_name": "TextOutput",
    "value": " The",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " capital",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " of",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " France",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " is",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " Paris",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": ".",
    "is_input": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "latency_ms": 0,
  },
  {
    "class_name": "TextOutput",
    "value": " It",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " is",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " not",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " only",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " the",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " largest",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " city",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " in",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " France",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " but",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " also",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " one",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " of",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " the",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " most",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " important",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " cultural",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " and",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " commercial",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " cent",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": "ers",
    "is_input": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "latency_ms": 0,
  },
  {
    "class_name": "TextOutput",
    "value": " in",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  },
  {
    "class_name": "TextOutput",
    "value": " Europe",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "latency_ms": 0
  }
];
