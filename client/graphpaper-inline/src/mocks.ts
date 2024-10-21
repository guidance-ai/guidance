// Mocks for interactive testing

import {type TextOutput, type RoleOpenerInput, type RoleCloserInput, type GenTokenExtra} from './stitch';

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
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": "What is the capital of France?",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
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
    "token_count": 0,
    "prob": 0
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
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " The",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " capital",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " of",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " France",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " is",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " Paris",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": ".",
    "is_input": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " It",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " is",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " not",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " only",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " the",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " largest",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " city",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " in",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " France",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " but",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " also",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " one",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " of",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " the",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " most",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " important",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " cultural",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " and",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " commercial",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " cent",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": "ers",
    "is_input": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " in",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " Europe",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  }
];

export const mockGenTokens: Array<GenTokenExtra> = [
  {
    "token_id": 32010,
    "prob": 0.0001471998548367992,
    "text": "<|user|>",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 29906,
        "prob": 0.11696617305278778,
        "text": "2",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29896,
        "prob": 0.10765588283538818,
        "text": "1",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29941,
        "prob": 0.05305387079715729,
        "text": "3",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 13,
        "prob": 0.034597232937812805,
        "text": "\n",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29946,
        "prob": 0.029241180047392845,
        "text": "4",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32010,
        "prob": 0.0001471998548367992,
        "text": "<|user|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 5618,
    "prob": 0.0000025939498300431296,
    "text": "\nWhat",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 306,
        "prob": 0.07540036737918854,
        "text": "I",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 512,
        "prob": 0.05664486810564995,
        "text": "In",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 448,
        "prob": 0.05062797665596008,
        "text": "-",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1724,
        "prob": 0.028206998482346535,
        "text": "What",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 6527,
        "prob": 0.024746403098106384,
        "text": "Could",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 5618,
        "prob": 0.0000025939498300431296,
        "text": "What",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 338,
    "prob": 0.39458176493644714,
    "text": " is",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 338,
        "prob": 0.39458176493644714,
        "text": "is",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 526,
        "prob": 0.20252063870429993,
        "text": "are",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29915,
        "prob": 0.03513995558023453,
        "text": "'",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 947,
        "prob": 0.029331203550100327,
        "text": "does",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 723,
        "prob": 0.0216793455183506,
        "text": "would",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 278,
    "prob": 0.8396025896072388,
    "text": " the",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 278,
        "prob": 0.8396025896072388,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 263,
        "prob": 0.04767822101712227,
        "text": "a",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 596,
        "prob": 0.014672422781586647,
        "text": "your",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 385,
        "prob": 0.01002463512122631,
        "text": "an",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29871,
        "prob": 0.002904066350311041,
        "text": "",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 7483,
    "prob": 0.00797582883387804,
    "text": " capital",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 995,
        "prob": 0.1165241077542305,
        "text": "value",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2533,
        "prob": 0.041466087102890015,
        "text": "sum",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 6297,
        "prob": 0.026729455217719078,
        "text": "role",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3001,
        "prob": 0.026081698015332222,
        "text": "total",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 26002,
        "prob": 0.025236951187253,
        "text": "significance",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 7483,
        "prob": 0.00797582883387804,
        "text": "capital",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 310,
    "prob": 0.6770157217979431,
    "text": " of",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 310,
        "prob": 0.6770157217979431,
        "text": "of",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4272,
        "prob": 0.3132016360759735,
        "text": "city",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 330,
        "prob": 0.0018248602282255888,
        "text": "g",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2133,
        "prob": 0.0009512827382422984,
        "text": "ization",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1891,
        "prob": 0.0006871097721159458,
        "text": "ized",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 3444,
    "prob": 0.8318758606910706,
    "text": " France",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 3444,
        "prob": 0.8318758606910706,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 7400,
        "prob": 0.03954508528113365,
        "text": "Canada",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 8314,
        "prob": 0.018492599949240685,
        "text": "Australia",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4810,
        "prob": 0.0115347346290946,
        "text": "CO",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 15198,
        "prob": 0.00963549967855215,
        "text": "Kaz",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 29973,
    "prob": 0.9806775450706482,
    "text": "?",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 29973,
        "prob": 0.9806775450706482,
        "text": "?",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29892,
        "prob": 0.008077787235379219,
        "text": ",",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 322,
        "prob": 0.004824829287827015,
        "text": "and",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.0023883178364485502,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 13,
        "prob": 0.001062076073139906,
        "text": "\n",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 32007,
    "prob": 0.2750667631626129,
    "text": "<|end|>",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 13,
        "prob": 0.4481906592845917,
        "text": "\n",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.2750667631626129,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29871,
        "prob": 0.1448775678873062,
        "text": "",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 9133,
        "prob": 0.014615979976952076,
        "text": "Prov",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 259,
        "prob": 0.011606425978243351,
        "text": " ",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 32001,
    "prob": 0.8795124888420105,
    "text": "\n<|assistant|>",
    "latency_ms": 0,
    "is_masked": false,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token_id": 32001,
        "prob": 0.8795124888420105,
        "text": "<|assistant|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32010,
        "prob": 0.04064176231622696,
        "text": "<|user|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 835,
        "prob": 0.030805600807070732,
        "text": "###",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3579,
        "prob": 0.017812293022871017,
        "text": "**",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 444,
        "prob": 0.014047935605049133,
        "text": "##",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 1576,
    "prob": 0.00006422422302421182,
    "text": "\n The",
    "latency_ms": 791.1298274993896,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 450,
        "prob": 0.9857171773910522,
        "text": "The",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3681,
        "prob": 0.011184466071426868,
        "text": "Paris",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3444,
        "prob": 0.0011331287678331137,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3579,
        "prob": 0.0004644235013984144,
        "text": "**",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 448,
        "prob": 0.00026546273147687316,
        "text": "-",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1576,
        "prob": 0.00006422422302421182,
        "text": "The",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 7483,
    "prob": 0.999437153339386,
    "text": " capital",
    "latency_ms": 187.8337860107422,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 7483,
        "prob": 0.999437153339386,
        "text": "capital",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1234,
        "prob": 0.0003310518804937601,
        "text": "answer",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1139,
        "prob": 0.00011426454148022458,
        "text": "question",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 6221,
        "prob": 0.00001873375367722474,
        "text": "official",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1857,
        "prob": 0.000014494446986645926,
        "text": "current",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 310,
    "prob": 0.9986506104469299,
    "text": " of",
    "latency_ms": 183.6080551147461,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 310,
        "prob": 0.9986506104469299,
        "text": "of",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4272,
        "prob": 0.0013466310920193791,
        "text": "city",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.0000010585043810351635,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29871,
        "prob": 6.448343583542737e-7,
        "text": "",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4412,
        "prob": 1.6180513284780318e-7,
        "text": "City",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 3444,
    "prob": 0.999962329864502,
    "text": " France",
    "latency_ms": 186.737060546875,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 3444,
        "prob": 0.999962329864502,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.000017144215235020965,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 278,
        "prob": 0.000008596942279837094,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3579,
        "prob": 0.000004998890744900564,
        "text": "**",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 16066,
        "prob": 0.0000016594119642832084,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 338,
    "prob": 0.9999902248382568,
    "text": " is",
    "latency_ms": 185.22119522094727,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 338,
        "prob": 0.9999902248382568,
        "text": "is",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29892,
        "prob": 0.000004214996351947775,
        "text": ",",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1752,
        "prob": 9.97037091110542e-7,
        "text": "ist",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29901,
        "prob": 8.132024618134892e-7,
        "text": ":",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 7.06779985648609e-7,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 3681,
    "prob": 0.9999117851257324,
    "text": " Paris",
    "latency_ms": 187.99996376037598,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 3681,
        "prob": 0.9999117851257324,
        "text": "Paris",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3579,
        "prob": 0.000049396130634704605,
        "text": "**",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1459,
        "prob": 0.000025039666070370004,
        "text": "Par",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.0000015687925269958214,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4257,
        "prob": 0.0000014064415836401167,
        "text": "named",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 29889,
    "prob": 0.9989786148071289,
    "text": ".",
    "latency_ms": 184.83901023864746,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 29889,
        "prob": 0.9989786148071289,
        "text": ".",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29892,
        "prob": 0.000967636238783598,
        "text": ",",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.0000353920622728765,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 313,
        "prob": 0.000010462333193572704,
        "text": "(",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29936,
        "prob": 0.0000014101174201641697,
        "text": ";",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 739,
    "prob": 0.3691301941871643,
    "text": " It",
    "latency_ms": 182.1460723876953,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 32007,
        "prob": 0.4036679267883301,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 739,
        "prob": 0.3691301941871643,
        "text": "It",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3681,
        "prob": 0.18517325818538666,
        "text": "Paris",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 5976,
        "prob": 0.009475378319621086,
        "text": "Loc",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 910,
        "prob": 0.007770691066980362,
        "text": "This",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 338,
    "prob": 0.7372497916221619,
    "text": " is",
    "latency_ms": 186.75732612609863,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 338,
        "prob": 0.7372497916221619,
        "text": "is",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29915,
        "prob": 0.22102417051792145,
        "text": "'",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 19700,
        "prob": 0.018399816006422043,
        "text": "serves",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 30010,
        "prob": 0.010582340881228447,
        "text": "’",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 756,
        "prob": 0.010219420306384563,
        "text": "has",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 451,
    "prob": 0.5775279402732849,
    "text": " not",
    "latency_ms": 192.30103492736816,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 451,
        "prob": 0.5775279402732849,
        "text": "not",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 278,
        "prob": 0.30363428592681885,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 884,
        "prob": 0.055202122777700424,
        "text": "also",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1716,
        "prob": 0.01739770546555519,
        "text": "both",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2998,
        "prob": 0.014207991771399975,
        "text": "known",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 871,
    "prob": 0.9856061339378357,
    "text": " only",
    "latency_ms": 186.07521057128906,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 871,
        "prob": 0.9856061339378357,
        "text": "only",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 925,
        "prob": 0.01367239747196436,
        "text": "just",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 13586,
        "prob": 0.0006267011631280184,
        "text": "merely",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3763,
        "prob": 0.00005819142097607255,
        "text": "simply",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2197,
        "prob": 0.000014317490240500774,
        "text": "ably",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 278,
    "prob": 0.982492208480835,
    "text": " the",
    "latency_ms": 203.89771461486816,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 278,
        "prob": 0.982492208480835,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3444,
        "prob": 0.009905748069286346,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 263,
        "prob": 0.006513569038361311,
        "text": "a",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1737,
        "prob": 0.00027604380738921463,
        "text": "ge",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2998,
        "prob": 0.0002052166237263009,
        "text": "known",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 10150,
    "prob": 0.4096444845199585,
    "text": " largest",
    "latency_ms": 196.5487003326416,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 4234,
        "prob": 0.4371485412120819,
        "text": "country",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 10150,
        "prob": 0.4096444845199585,
        "text": "largest",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 8604,
        "prob": 0.08317449688911438,
        "text": "political",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1556,
        "prob": 0.03200649842619896,
        "text": "most",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 7483,
        "prob": 0.02671337127685547,
        "text": "capital",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 4272,
    "prob": 0.999167799949646,
    "text": " city",
    "latency_ms": 307.5690269470215,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 4272,
        "prob": 0.999167799949646,
        "text": "city",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 25311,
        "prob": 0.0005337278707884252,
        "text": "metropol",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 310,
        "prob": 0.00022632069885730743,
        "text": "of",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 322,
        "prob": 0.000033229585824301466,
        "text": "and",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 5176,
        "prob": 0.000019576928025344387,
        "text": "French",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 297,
    "prob": 0.6239914894104004,
    "text": " in",
    "latency_ms": 306.05530738830566,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 297,
        "prob": 0.6239914894104004,
        "text": "in",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 310,
        "prob": 0.2546098828315735,
        "text": "of",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 541,
        "prob": 0.115852952003479,
        "text": "but",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2629,
        "prob": 0.0023302428890019655,
        "text": "within",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 373,
        "prob": 0.0022443863563239574,
        "text": "on",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 3444,
    "prob": 0.6372988224029541,
    "text": " France",
    "latency_ms": 179.40211296081543,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 3444,
        "prob": 0.6372988224029541,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 278,
        "prob": 0.3625364601612091,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 25311,
        "prob": 0.00004971169983036816,
        "text": "metropol",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4958,
        "prob": 0.000036445129808271304,
        "text": "terms",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4234,
        "prob": 0.00002782431874948088,
        "text": "country",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 541,
    "prob": 0.9793675541877747,
    "text": " but",
    "latency_ms": 178.9560317993164,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 541,
        "prob": 0.9793675541877747,
        "text": "but",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29892,
        "prob": 0.02007644809782505,
        "text": ",",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 411,
        "prob": 0.00042435049545019865,
        "text": "with",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29936,
        "prob": 0.00003524085332173854,
        "text": ";",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 322,
        "prob": 0.000032845084206201136,
        "text": "and",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 884,
    "prob": 0.9989857077598572,
    "text": " also",
    "latency_ms": 181.1509132385254,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 884,
        "prob": 0.9989857077598572,
        "text": "also",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 338,
        "prob": 0.000602494168560952,
        "text": "is",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 372,
        "prob": 0.0001398069434799254,
        "text": "it",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 756,
        "prob": 0.00009900984878186136,
        "text": "has",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 19700,
        "prob": 0.00004597084625856951,
        "text": "serves",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 697,
    "prob": 0.3192960321903229,
    "text": " one",
    "latency_ms": 181.49805068969727,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 697,
        "prob": 0.3192960321903229,
        "text": "one",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 263,
        "prob": 0.3170034885406494,
        "text": "a",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 19700,
        "prob": 0.16621005535125732,
        "text": "serves",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 278,
        "prob": 0.07940118759870529,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 967,
        "prob": 0.06430594623088837,
        "text": "its",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 310,
    "prob": 0.9997043013572693,
    "text": " of",
    "latency_ms": 186.09094619750977,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 310,
        "prob": 0.9997043013572693,
        "text": "of",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 411,
        "prob": 0.00016257438983302563,
        "text": "with",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 393,
        "prob": 0.00008909530879464,
        "text": "that",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 32007,
        "prob": 0.000011212241588509642,
        "text": "<|end|>",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2998,
        "prob": 0.000008543695912521798,
        "text": "known",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 278,
    "prob": 0.9882127642631531,
    "text": " the",
    "latency_ms": 183.80475044250488,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 278,
        "prob": 0.9882127642631531,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 967,
        "prob": 0.010567160323262215,
        "text": "its",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4092,
        "prob": 0.0011446235002949834,
        "text": "Europe",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3444,
        "prob": 0.00002321232750546187,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1009,
        "prob": 0.000013299547390488442,
        "text": "their",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 1556,
    "prob": 0.8675132989883423,
    "text": " most",
    "latency_ms": 189.83983993530273,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 1556,
        "prob": 0.8675132989883423,
        "text": "most",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 4655,
        "prob": 0.07431504875421524,
        "text": "major",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 363,
        "prob": 0.03438625484704971,
        "text": "for",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3186,
        "prob": 0.012852047570049763,
        "text": "world",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 8236,
        "prob": 0.006348212715238333,
        "text": "leading",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 4100,
    "prob": 0.5192220211029053,
    "text": " important",
    "latency_ms": 182.08789825439453,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 4100,
        "prob": 0.5192220211029053,
        "text": "important",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 9849,
        "prob": 0.2267724722623825,
        "text": "icon",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 13834,
        "prob": 0.10495571792125702,
        "text": "famous",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 16669,
        "prob": 0.07289287447929382,
        "text": "visited",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 14938,
        "prob": 0.01952418126165867,
        "text": "popul",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 16375,
    "prob": 0.7161048054695129,
    "text": " cultural",
    "latency_ms": 181.44798278808594,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 16375,
        "prob": 0.7161048054695129,
        "text": "cultural",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 322,
        "prob": 0.134635791182518,
        "text": "and",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 5534,
        "prob": 0.056849200278520584,
        "text": "global",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 17407,
        "prob": 0.04618300125002861,
        "text": "economic",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 8604,
        "prob": 0.016588550060987473,
        "text": "political",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 322,
    "prob": 0.8125864863395691,
    "text": " and",
    "latency_ms": 182.1150779724121,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 322,
        "prob": 0.8125864863395691,
        "text": "and",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29892,
        "prob": 0.15330319106578827,
        "text": ",",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1644,
        "prob": 0.02392854355275631,
        "text": "cent",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 19766,
        "prob": 0.009282657876610756,
        "text": "hub",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 14368,
        "prob": 0.0005479487008415163,
        "text": "cities",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 12128,
    "prob": 0.6127476096153259,
    "text": " commercial",
    "latency_ms": 213.25373649597168,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 12128,
        "prob": 0.6127476096153259,
        "text": "commercial",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 17407,
        "prob": 0.26235485076904297,
        "text": "economic",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 5381,
        "prob": 0.0556219108402729,
        "text": "business",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 15839,
        "prob": 0.050407830625772476,
        "text": "historical",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 18161,
        "prob": 0.010083277709782124,
        "text": "financial",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 1644,
    "prob": 0.9685200452804565,
    "text": " cent",
    "latency_ms": 187.485933303833,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 1644,
        "prob": 0.9685200452804565,
        "text": "cent",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 19766,
        "prob": 0.028621848672628403,
        "text": "hub",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 14368,
        "prob": 0.0017020675586536527,
        "text": "cities",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 11934,
        "prob": 0.0007980230147950351,
        "text": "capit",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 25311,
        "prob": 0.00012140109174652025,
        "text": "metropol",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 414,
    "prob": 0.9984211921691895,
    "text": "ers",
    "latency_ms": 191.87593460083008,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 414,
        "prob": 0.9984211921691895,
        "text": "ers",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 690,
        "prob": 0.0015741386450827122,
        "text": "res",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29878,
        "prob": 0.0000014762757700736984,
        "text": "r",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1007,
        "prob": 0.0000012054251783411019,
        "text": "roid",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 2288,
        "prob": 0.0000010710768947319593,
        "text": "rs",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 297,
    "prob": 0.9748722314834595,
    "text": " in",
    "latency_ms": 183.5949420928955,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 297,
        "prob": 0.9748722314834595,
        "text": "in",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 13149,
        "prob": 0.011514736339449883,
        "text": "glob",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 29889,
        "prob": 0.007136909756809473,
        "text": ".",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 310,
        "prob": 0.0027969765942543745,
        "text": "of",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 451,
        "prob": 0.0016226732404902577,
        "text": "not",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  },
  {
    "token_id": 4092,
    "prob": 0.49644505977630615,
    "text": " Europe",
    "latency_ms": 181.49399757385254,
    "is_masked": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token_id": 4092,
        "prob": 0.49644505977630615,
        "text": "Europe",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 278,
        "prob": 0.4951532781124115,
        "text": "the",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 3444,
        "prob": 0.005787685979157686,
        "text": "France",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 10504,
        "prob": 0.0018548371735960245,
        "text": "Western",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      },
      {
        "token_id": 1716,
        "prob": 0.0003810191119555384,
        "text": "both",
        "latency_ms": 0,
        "is_masked": false,
        "is_generated": false,
        "is_force_forwarded": false,
        "is_input": false
      }
    ]
  }
];