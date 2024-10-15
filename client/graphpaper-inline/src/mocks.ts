// Mocks for interactive testing

import {type TextOutput, type RoleOpenerInput, type RoleCloserInput, type GenToken} from './stitch';

export const mockNodeAttrs: Array<RoleCloserInput | RoleOpenerInput | TextOutput> = [
  {
    "class_name": "RoleOpenerInput",
    "name": "system",
    "text": "<|im_start|>system\n",
    "closer_text": "<|im_end|>\n"
  },
  {
    "class_name": "TextOutput",
    "value": "<|im_start|>system\n",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": "You are responsible for autocompleting a sentence.",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "RoleCloserInput",
    "name": "system",
    "text": "<|im_end|>\n"
  },
  {
    "class_name": "TextOutput",
    "value": "<|im_end|>\n",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "RoleOpenerInput",
    "name": "user",
    "text": "<|im_start|>user\n",
    "closer_text": "<|im_end|>\n"
  },
  {
    "class_name": "TextOutput",
    "value": "<|im_start|>user\n",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": "Roses are red and ",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": "you",
    "is_input": false,
    "is_generated": false,
    "is_force_forwarded": true,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " are",
    "is_input": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": " responsible",
    "is_input": false,
    "is_generated": true,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "RoleCloserInput",
    "name": "user",
    "text": "<|im_end|>\n"
  },
  {
    "class_name": "TextOutput",
    "value": "<|im_end|>\n",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  },
  {
    "class_name": "TextOutput",
    "value": ">:(",
    "is_input": true,
    "is_generated": false,
    "is_force_forwarded": false,
    "token_count": 0,
    "prob": 0
  }
];

export const mockGenTokens: Array<GenToken> = [
  {
    "token": 27,
    "prob": 1,
    "text": "<",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 27,
        "prob": 1,
        "text": "<",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.0008722934871912003,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 198,
        "prob": 0.029958423227071762,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.013489649631083012,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 16,
        "prob": 0.011806909926235676,
        "text": "1",
        "is_masked": false
      },
      {
        "token": 12,
        "prob": 0.010169479995965958,
        "text": "-",
        "is_masked": false
      },
      {
        "token": 15,
        "prob": 0.009271609596908092,
        "text": "0",
        "is_masked": false
      },
      {
        "token": 91,
        "prob": 0.0008722934871912003,
        "text": "|",
        "is_masked": false
      }
    ]
  },
  {
    "token": 320,
    "prob": 0.000027396810764912516,
    "text": "im",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 930,
        "prob": 0.12004928290843964,
        "text": " |",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.06939840316772461,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.03129403665661812,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 198,
        "prob": 0.028110388666391373,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 60,
        "prob": 0.02136152982711792,
        "text": "]",
        "is_masked": false
      },
      {
        "token": 320,
        "prob": 0.000027396810764912516,
        "text": "im",
        "is_masked": false
      }
    ]
  },
  {
    "token": 62,
    "prob": 0.01482831034809351,
    "text": "_",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.42150357365608215,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 7629,
        "prob": 0.023308370262384415,
        "text": "perial",
        "is_masked": false
      },
      {
        "token": 60,
        "prob": 0.020157789811491966,
        "text": "]",
        "is_masked": false
      },
      {
        "token": 14,
        "prob": 0.018618104979395866,
        "text": "/",
        "is_masked": false
      },
      {
        "token": 92,
        "prob": 0.015273063443601131,
        "text": "}",
        "is_masked": false
      },
      {
        "token": 62,
        "prob": 0.01482831034809351,
        "text": "_",
        "is_masked": false
      }
    ]
  },
  {
    "token": 9688,
    "prob": 0.0003535318828653544,
    "text": "start",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.017649956047534943,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 79,
        "prob": 0.011138186790049076,
        "text": "p",
        "is_masked": false
      },
      {
        "token": 16366,
        "prob": 0.011045439168810844,
        "text": "client",
        "is_masked": false
      },
      {
        "token": 82,
        "prob": 0.008266090415418148,
        "text": "s",
        "is_masked": false
      },
      {
        "token": 15908,
        "prob": 0.00811094231903553,
        "text": "dir",
        "is_masked": false
      },
      {
        "token": 9688,
        "prob": 0.0003535318828653544,
        "text": "start",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.12612135708332062,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 62,
        "prob": 0.3673606216907501,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 91,
        "prob": 0.12612135708332062,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 929,
        "prob": 0.041732169687747955,
        "text": "up",
        "is_masked": false
      },
      {
        "token": 92,
        "prob": 0.032134898006916046,
        "text": "}",
        "is_masked": false
      },
      {
        "token": 28,
        "prob": 0.024743985384702682,
        "text": "=",
        "is_masked": false
      }
    ]
  },
  {
    "token": 29,
    "prob": 0.012729871086776257,
    "text": ">",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 259,
        "prob": 0.028011340647935867,
        "text": "in",
        "is_masked": false
      },
      {
        "token": 4023,
        "prob": 0.01824311725795269,
        "text": "http",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.015950094908475876,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 929,
        "prob": 0.01563503034412861,
        "text": "up",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.012729871086776257,
        "text": ">",
        "is_masked": false
      }
    ]
  },
  {
    "token": 10057,
    "prob": 0.00004459881893126294,
    "text": "system",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.2632869482040405,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 198,
        "prob": 0.05016319826245308,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 1279,
        "prob": 0.04930492863059044,
        "text": " <",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.045030541718006134,
        "text": " |",
        "is_masked": false
      },
      {
        "token": 28,
        "prob": 0.023192651569843292,
        "text": "=",
        "is_masked": false
      },
      {
        "token": 10057,
        "prob": 0.00004459881893126294,
        "text": "system",
        "is_masked": false
      }
    ]
  },
  {
    "token": 198,
    "prob": 0.01464146375656128,
    "text": "\n",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.20154249668121338,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 3556,
        "prob": 0.11300163716077805,
        "text": "</",
        "is_masked": false
      },
      {
        "token": 62,
        "prob": 0.11157143115997314,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 14,
        "prob": 0.08955718576908112,
        "text": "/",
        "is_masked": false
      },
      {
        "token": 12,
        "prob": 0.06915956735610962,
        "text": "-",
        "is_masked": false
      },
      {
        "token": 198,
        "prob": 0.01464146375656128,
        "text": "\n",
        "is_masked": false
      }
    ]
  },
  {
    "token": 1639,
    "prob": 2.8333360546639597e-7,
    "text": "You",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 198,
        "prob": 0.9993786811828613,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.00007136884960345924,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 1,
        "prob": 0.00004172701665083878,
        "text": "\"",
        "is_masked": false
      },
      {
        "token": 25,
        "prob": 0.0000411842847825028,
        "text": ":",
        "is_masked": false
      },
      {
        "token": 11,
        "prob": 0.000037523903301917017,
        "text": ",",
        "is_masked": false
      },
      {
        "token": 1639,
        "prob": 2.8333360546639597e-7,
        "text": "You",
        "is_masked": false
      }
    ]
  },
  {
    "token": 389,
    "prob": 0.04107223451137543,
    "text": " are",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 460,
        "prob": 0.25261735916137695,
        "text": " can",
        "is_masked": false
      },
      {
        "token": 815,
        "prob": 0.05772041901946068,
        "text": " should",
        "is_masked": false
      },
      {
        "token": 821,
        "prob": 0.04306665435433388,
        "text": "'re",
        "is_masked": false
      },
      {
        "token": 1183,
        "prob": 0.04206516593694687,
        "text": "'ll",
        "is_masked": false
      },
      {
        "token": 389,
        "prob": 0.04107223451137543,
        "text": " are",
        "is_masked": false
      }
    ]
  },
  {
    "token": 4497,
    "prob": 0.0011660599848255515,
    "text": " responsible",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 783,
        "prob": 0.09382512420415878,
        "text": " now",
        "is_masked": false
      },
      {
        "token": 407,
        "prob": 0.06339426338672638,
        "text": " not",
        "is_masked": false
      },
      {
        "token": 18832,
        "prob": 0.044431913644075394,
        "text": " logged",
        "is_masked": false
      },
      {
        "token": 1262,
        "prob": 0.03386500105261803,
        "text": " using",
        "is_masked": false
      },
      {
        "token": 287,
        "prob": 0.02802243083715439,
        "text": " in",
        "is_masked": false
      },
      {
        "token": 4497,
        "prob": 0.0011660599848255515,
        "text": " responsible",
        "is_masked": false
      }
    ]
  },
  {
    "token": 329,
    "prob": 0.9461202621459961,
    "text": " for",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 329,
        "prob": 0.9461202621459961,
        "text": " for",
        "is_masked": false
      },
      {
        "token": 284,
        "prob": 0.023375902324914932,
        "text": " to",
        "is_masked": false
      },
      {
        "token": 286,
        "prob": 0.003662876319140196,
        "text": " of",
        "is_masked": false
      },
      {
        "token": 691,
        "prob": 0.00276382127776742,
        "text": " only",
        "is_masked": false
      },
      {
        "token": 11,
        "prob": 0.002147114370018244,
        "text": ",",
        "is_masked": false
      }
    ]
  },
  {
    "token": 1960,
    "prob": 0.000013145288903615437,
    "text": " aut",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 262,
        "prob": 0.05765107646584511,
        "text": " the",
        "is_masked": false
      },
      {
        "token": 534,
        "prob": 0.04888231307268143,
        "text": " your",
        "is_masked": false
      },
      {
        "token": 13359,
        "prob": 0.041589509695768356,
        "text": " ensuring",
        "is_masked": false
      },
      {
        "token": 10941,
        "prob": 0.029673287644982338,
        "text": " maintaining",
        "is_masked": false
      },
      {
        "token": 4634,
        "prob": 0.023319991305470467,
        "text": " setting",
        "is_masked": false
      },
      {
        "token": 1960,
        "prob": 0.000013145288903615437,
        "text": " aut",
        "is_masked": false
      }
    ]
  },
  {
    "token": 42829,
    "prob": 0.10221736133098602,
    "text": "ocom",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 36221,
        "prob": 0.1104634627699852,
        "text": "ocon",
        "is_masked": false
      },
      {
        "token": 42829,
        "prob": 0.10221736133098602,
        "text": "ocom",
        "is_masked": false
      },
      {
        "token": 349,
        "prob": 0.08124707639217377,
        "text": "ol",
        "is_masked": false
      },
      {
        "token": 420,
        "prob": 0.0743418037891388,
        "text": "oc",
        "is_masked": false
      },
      {
        "token": 375,
        "prob": 0.06453607976436615,
        "text": "od",
        "is_masked": false
      }
    ]
  },
  {
    "token": 47130,
    "prob": 0.1489240676164627,
    "text": "pleting",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 24547,
        "prob": 0.47585758566856384,
        "text": "pletion",
        "is_masked": false
      },
      {
        "token": 6677,
        "prob": 0.3272377550601959,
        "text": "plete",
        "is_masked": false
      },
      {
        "token": 47130,
        "prob": 0.1489240676164627,
        "text": "pleting",
        "is_masked": false
      },
      {
        "token": 37069,
        "prob": 0.010913231410086155,
        "text": "plet",
        "is_masked": false
      },
      {
        "token": 48074,
        "prob": 0.009519742801785469,
        "text": "puting",
        "is_masked": false
      }
    ]
  },
  {
    "token": 257,
    "prob": 0.02377660945057869,
    "text": " a",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 262,
        "prob": 0.3159867525100708,
        "text": " the",
        "is_masked": false
      },
      {
        "token": 428,
        "prob": 0.09937503933906555,
        "text": " this",
        "is_masked": false
      },
      {
        "token": 477,
        "prob": 0.06309774518013,
        "text": " all",
        "is_masked": false
      },
      {
        "token": 534,
        "prob": 0.04899457469582558,
        "text": " your",
        "is_masked": false
      },
      {
        "token": 597,
        "prob": 0.02953566052019596,
        "text": " any",
        "is_masked": false
      },
      {
        "token": 257,
        "prob": 0.02377660945057869,
        "text": " a",
        "is_masked": false
      }
    ]
  },
  {
    "token": 6827,
    "prob": 0.000513878301717341,
    "text": " sentence",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 3141,
        "prob": 0.05367027968168259,
        "text": " command",
        "is_masked": false
      },
      {
        "token": 3275,
        "prob": 0.043787699192762375,
        "text": " message",
        "is_masked": false
      },
      {
        "token": 1351,
        "prob": 0.034799974411726,
        "text": " list",
        "is_masked": false
      },
      {
        "token": 2393,
        "prob": 0.03282475844025612,
        "text": " file",
        "is_masked": false
      },
      {
        "token": 4876,
        "prob": 0.021464355289936066,
        "text": " task",
        "is_masked": false
      },
      {
        "token": 6827,
        "prob": 0.000513878301717341,
        "text": " sentence",
        "is_masked": false
      }
    ]
  },
  {
    "token": 29847,
    "prob": 0.00025674980133771896,
    "text": ".<",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 13,
        "prob": 0.18595099449157715,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 287,
        "prob": 0.08436127007007599,
        "text": " in",
        "is_masked": false
      },
      {
        "token": 11,
        "prob": 0.06681626290082932,
        "text": ",",
        "is_masked": false
      },
      {
        "token": 351,
        "prob": 0.05892467126250267,
        "text": " with",
        "is_masked": false
      },
      {
        "token": 326,
        "prob": 0.049389537423849106,
        "text": " that",
        "is_masked": false
      },
      {
        "token": 29847,
        "prob": 0.00025674980133771896,
        "text": ".<",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.6894029378890991,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.6894029378890991,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 1671,
        "prob": 0.11479752510786057,
        "text": "br",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.008212408982217312,
        "text": " |",
        "is_masked": false
      },
      {
        "token": 1652,
        "prob": 0.00741753913462162,
        "text": "lf",
        "is_masked": false
      },
      {
        "token": 65,
        "prob": 0.0065499767661094666,
        "text": "b",
        "is_masked": false
      }
    ]
  },
  {
    "token": 320,
    "prob": 0.3683234453201294,
    "text": "im",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 320,
        "prob": 0.3683234453201294,
        "text": "im",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.026579268276691437,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 361,
        "prob": 0.02362215332686901,
        "text": "if",
        "is_masked": false
      },
      {
        "token": 3546,
        "prob": 0.019490960985422134,
        "text": "Im",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.016572270542383194,
        "text": " |",
        "is_masked": false
      }
    ]
  },
  {
    "token": 62,
    "prob": 0.9976491332054138,
    "text": "_",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 62,
        "prob": 0.9976491332054138,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 91,
        "prob": 0.000336194148985669,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 41052,
        "prob": 0.00012316853099036962,
        "text": "_(",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.000117526447866112,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 59,
        "prob": 0.00009838600090006366,
        "text": "\\",
        "is_masked": false
      }
    ]
  },
  {
    "token": 437,
    "prob": 0.4462008774280548,
    "text": "end",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 437,
        "prob": 0.4462008774280548,
        "text": "end",
        "is_masked": false
      },
      {
        "token": 9688,
        "prob": 0.09380462765693665,
        "text": "start",
        "is_masked": false
      },
      {
        "token": 12957,
        "prob": 0.017503555864095688,
        "text": "last",
        "is_masked": false
      },
      {
        "token": 11338,
        "prob": 0.016530489549040794,
        "text": "stop",
        "is_masked": false
      },
      {
        "token": 27471,
        "prob": 0.012605799362063408,
        "text": "begin",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.9823351502418518,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.9823351502418518,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.002384455408900976,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 60,
        "prob": 0.0023481370881199837,
        "text": "]",
        "is_masked": false
      },
      {
        "token": 15886,
        "prob": 0.002319696359336376,
        "text": "||",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.0013476606691256166,
        "text": " |",
        "is_masked": false
      }
    ]
  },
  {
    "token": 29,
    "prob": 0.9560039639472961,
    "text": ">",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 29,
        "prob": 0.9560039639472961,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 6927,
        "prob": 0.005681615322828293,
        "text": "><",
        "is_masked": false
      },
      {
        "token": 60,
        "prob": 0.0032080409582704306,
        "text": "]",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.002467683283612132,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 4211,
        "prob": 0.0021098603028804064,
        "text": ">>",
        "is_masked": false
      }
    ]
  },
  {
    "token": 198,
    "prob": 0.02139684557914734,
    "text": "\n",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 10057,
        "prob": 0.35756662487983704,
        "text": "system",
        "is_masked": false
      },
      {
        "token": 198,
        "prob": 0.02139684557914734,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 11964,
        "prob": 0.01934202015399933,
        "text": "System",
        "is_masked": false
      },
      {
        "token": 1662,
        "prob": 0.010047300718724728,
        "text": "not",
        "is_masked": false
      },
      {
        "token": 1169,
        "prob": 0.006138301454484463,
        "text": "the",
        "is_masked": false
      }
    ]
  },
  {
    "token": 27,
    "prob": 0.039419274777173996,
    "text": "<",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 1639,
        "prob": 0.1665498912334442,
        "text": "You",
        "is_masked": false
      },
      {
        "token": 464,
        "prob": 0.06005862355232239,
        "text": "The",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.039419274777173996,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 1212,
        "prob": 0.027387971058487892,
        "text": "This",
        "is_masked": false
      },
      {
        "token": 1532,
        "prob": 0.022758493199944496,
        "text": "If",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.7559300065040588,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.7559300065040588,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 320,
        "prob": 0.006860977970063686,
        "text": "im",
        "is_masked": false
      },
      {
        "token": 15886,
        "prob": 0.0064566307701170444,
        "text": "||",
        "is_masked": false
      },
      {
        "token": 1671,
        "prob": 0.004361273255199194,
        "text": "br",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.004342366941273212,
        "text": " |",
        "is_masked": false
      }
    ]
  },
  {
    "token": 320,
    "prob": 0.8291174173355103,
    "text": "im",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 320,
        "prob": 0.8291174173355103,
        "text": "im",
        "is_masked": false
      },
      {
        "token": 3546,
        "prob": 0.012388699688017368,
        "text": "Im",
        "is_masked": false
      },
      {
        "token": 361,
        "prob": 0.007399916183203459,
        "text": "if",
        "is_masked": false
      },
      {
        "token": 72,
        "prob": 0.004467112943530083,
        "text": "i",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.0026658582501113415,
        "text": ">",
        "is_masked": false
      }
    ]
  },
  {
    "token": 62,
    "prob": 0.9993433356285095,
    "text": "_",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 62,
        "prob": 0.9993433356285095,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 7,
        "prob": 0.0000683960533933714,
        "text": "(",
        "is_masked": false
      },
      {
        "token": 59,
        "prob": 0.00006206566467881203,
        "text": "\\",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.00004414364229887724,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.00002257493724755477,
        "text": ">",
        "is_masked": false
      }
    ]
  },
  {
    "token": 9688,
    "prob": 0.6440587043762207,
    "text": "start",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 9688,
        "prob": 0.6440587043762207,
        "text": "start",
        "is_masked": false
      },
      {
        "token": 437,
        "prob": 0.05475924164056778,
        "text": "end",
        "is_masked": false
      },
      {
        "token": 27471,
        "prob": 0.043713200837373734,
        "text": "begin",
        "is_masked": false
      },
      {
        "token": 11338,
        "prob": 0.003659052774310112,
        "text": "stop",
        "is_masked": false
      },
      {
        "token": 38690,
        "prob": 0.003096950938925147,
        "text": "starting",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.9499432444572449,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.9499432444572449,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 62,
        "prob": 0.00848131813108921,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 15886,
        "prob": 0.005286256782710552,
        "text": "||",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.003634629538282752,
        "text": " |",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.0032877251505851746,
        "text": ">",
        "is_masked": false
      }
    ]
  },
  {
    "token": 29,
    "prob": 0.9690060615539551,
    "text": ">",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 29,
        "prob": 0.9690060615539551,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 6927,
        "prob": 0.004052347969263792,
        "text": "><",
        "is_masked": false
      },
      {
        "token": 24618,
        "prob": 0.0019232694758102298,
        "text": ">\"",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.0012727173743769526,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 4211,
        "prob": 0.0011409870348870754,
        "text": ">>",
        "is_masked": false
      }
    ]
  },
  {
    "token": 7220,
    "prob": 0.003625249955803156,
    "text": "user",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 10057,
        "prob": 0.43270590901374817,
        "text": "system",
        "is_masked": false
      },
      {
        "token": 11964,
        "prob": 0.012400631792843342,
        "text": "System",
        "is_masked": false
      },
      {
        "token": 9688,
        "prob": 0.008726297877728939,
        "text": "start",
        "is_masked": false
      },
      {
        "token": 847,
        "prob": 0.007208186201751232,
        "text": "other",
        "is_masked": false
      },
      {
        "token": 198,
        "prob": 0.006247976794838905,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 7220,
        "prob": 0.003625249955803156,
        "text": "user",
        "is_masked": false
      }
    ]
  },
  {
    "token": 198,
    "prob": 0.5544214248657227,
    "text": "\n",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 198,
        "prob": 0.5544214248657227,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 62,
        "prob": 0.04264130815863609,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 12,
        "prob": 0.029624110087752342,
        "text": "-",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.014048474840819836,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 14,
        "prob": 0.011965525336563587,
        "text": "/",
        "is_masked": false
      }
    ]
  },
  {
    "token": 49,
    "prob": 0.00022193520271684974,
    "text": "R",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 1639,
        "prob": 0.39992958307266235,
        "text": "You",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.11402493715286255,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 464,
        "prob": 0.05199533700942993,
        "text": "The",
        "is_masked": false
      },
      {
        "token": 1212,
        "prob": 0.01731906458735466,
        "text": "This",
        "is_masked": false
      },
      {
        "token": 3556,
        "prob": 0.016215169802308083,
        "text": "</",
        "is_masked": false
      },
      {
        "token": 49,
        "prob": 0.00022193520271684974,
        "text": "R",
        "is_masked": false
      }
    ]
  },
  {
    "token": 4629,
    "prob": 0.0011175252730026841,
    "text": "oses",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 12453,
        "prob": 0.20455273985862732,
        "text": "ename",
        "is_masked": false
      },
      {
        "token": 964,
        "prob": 0.08992191404104233,
        "text": "ever",
        "is_masked": false
      },
      {
        "token": 864,
        "prob": 0.06109633296728134,
        "text": "ational",
        "is_masked": false
      },
      {
        "token": 3798,
        "prob": 0.05252346768975258,
        "text": "esc",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.02254168502986431,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 4629,
        "prob": 0.0011175252730026841,
        "text": "oses",
        "is_masked": false
      }
    ]
  },
  {
    "token": 389,
    "prob": 0.13565655052661896,
    "text": " are",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 389,
        "prob": 0.13565655052661896,
        "text": " are",
        "is_masked": false
      },
      {
        "token": 262,
        "prob": 0.03858645260334015,
        "text": " the",
        "is_masked": false
      },
      {
        "token": 290,
        "prob": 0.027977442368865013,
        "text": " and",
        "is_masked": false
      },
      {
        "token": 481,
        "prob": 0.027524998411536217,
        "text": " will",
        "is_masked": false
      },
      {
        "token": 345,
        "prob": 0.025067530572414398,
        "text": " you",
        "is_masked": false
      }
    ]
  },
  {
    "token": 2266,
    "prob": 0.0004926767433062196,
    "text": " red",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 407,
        "prob": 0.05439644306898117,
        "text": " not",
        "is_masked": false
      },
      {
        "token": 6338,
        "prob": 0.021079493686556816,
        "text": " automatically",
        "is_masked": false
      },
      {
        "token": 973,
        "prob": 0.018136756494641304,
        "text": " used",
        "is_masked": false
      },
      {
        "token": 257,
        "prob": 0.015725180506706238,
        "text": " a",
        "is_masked": false
      },
      {
        "token": 1464,
        "prob": 0.015290400944650173,
        "text": " always",
        "is_masked": false
      },
      {
        "token": 2266,
        "prob": 0.0004926767433062196,
        "text": " red",
        "is_masked": false
      }
    ]
  },
  {
    "token": 290,
    "prob": 0.04066063463687897,
    "text": " and",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 18156,
        "prob": 0.24244694411754608,
        "text": "efined",
        "is_masked": false
      },
      {
        "token": 5791,
        "prob": 0.14630955457687378,
        "text": "rawn",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.07232632488012314,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 11,
        "prob": 0.05644136667251587,
        "text": ",",
        "is_masked": false
      },
      {
        "token": 1831,
        "prob": 0.04790006950497627,
        "text": "raw",
        "is_masked": false
      },
      {
        "token": 290,
        "prob": 0.04066063463687897,
        "text": " and",
        "is_masked": false
      }
    ]
  },
  {
    "token": 345,
    "prob": 0.05122449994087219,
    "text": " you",
    "is_masked": false,
    "latency_ms": 1370.4283237457275,
    "is_generated": false,
    "is_force_forwarded": true,
    "is_input": false,
    "top_k": [
      {
        "token": 345,
        "prob": 0.05122449994087219,
        "text": " you",
        "is_masked": false
      },
      {
        "token": 262,
        "prob": 0.02627693861722946,
        "text": " the",
        "is_masked": true
      },
      {
        "token": 2266,
        "prob": 0.02242019586265087,
        "text": " red",
        "is_masked": true
      },
      {
        "token": 389,
        "prob": 0.01836078055202961,
        "text": " are",
        "is_masked": true
      },
      {
        "token": 2330,
        "prob": 0.016507919877767563,
        "text": " white",
        "is_masked": true
      }
    ]
  },
  {
    "token": 389,
    "prob": 0.2900826930999756,
    "text": " are",
    "is_masked": false,
    "latency_ms": 18.9211368560791,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token": 389,
        "prob": 0.2900826930999756,
        "text": " are",
        "is_masked": false
      },
      {
        "token": 1276,
        "prob": 0.10353875160217285,
        "text": " must",
        "is_masked": true
      },
      {
        "token": 460,
        "prob": 0.08833532780408859,
        "text": " can",
        "is_masked": true
      },
      {
        "token": 423,
        "prob": 0.06707955151796341,
        "text": " have",
        "is_masked": true
      },
      {
        "token": 761,
        "prob": 0.05822150409221649,
        "text": " need",
        "is_masked": true
      }
    ]
  },
  {
    "token": 4497,
    "prob": 0.43562158942222595,
    "text": " responsible",
    "is_masked": false,
    "latency_ms": 41.890621185302734,
    "is_generated": true,
    "is_force_forwarded": false,
    "is_input": false,
    "top_k": [
      {
        "token": 4497,
        "prob": 0.43562158942222595,
        "text": " responsible",
        "is_masked": false
      },
      {
        "token": 407,
        "prob": 0.04081293195486069,
        "text": " not",
        "is_masked": true
      },
      {
        "token": 262,
        "prob": 0.023834867402911186,
        "text": " the",
        "is_masked": true
      },
      {
        "token": 18832,
        "prob": 0.011552640236914158,
        "text": " logged",
        "is_masked": true
      },
      {
        "token": 257,
        "prob": 0.0088853994384408,
        "text": " a",
        "is_masked": true
      }
    ]
  },
  {
    "token": 27,
    "prob": 0.0004999017692171037,
    "text": "<",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 329,
        "prob": 0.9797589778900146,
        "text": " for",
        "is_masked": false
      },
      {
        "token": 284,
        "prob": 0.003256487660109997,
        "text": " to",
        "is_masked": false
      },
      {
        "token": 286,
        "prob": 0.001405500341206789,
        "text": " of",
        "is_masked": false
      },
      {
        "token": 1279,
        "prob": 0.001338837668299675,
        "text": " <",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.001244492013938725,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.0004999017692171037,
        "text": "<",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.9909712076187134,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.9909712076187134,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 15886,
        "prob": 0.0004430859407875687,
        "text": "||",
        "is_masked": false
      },
      {
        "token": 10,
        "prob": 0.00040545235970057547,
        "text": "+",
        "is_masked": false
      },
      {
        "token": 11139,
        "prob": 0.0003702065150719136,
        "text": "\\/",
        "is_masked": false
      },
      {
        "token": 930,
        "prob": 0.00035656234831549227,
        "text": " |",
        "is_masked": false
      }
    ]
  },
  {
    "token": 320,
    "prob": 0.9680559039115906,
    "text": "im",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 320,
        "prob": 0.9680559039115906,
        "text": "im",
        "is_masked": false
      },
      {
        "token": 3546,
        "prob": 0.0025652858894318342,
        "text": "Im",
        "is_masked": false
      },
      {
        "token": 72,
        "prob": 0.0016249691834673285,
        "text": "i",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.0013347472995519638,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 361,
        "prob": 0.0009765964932739735,
        "text": "if",
        "is_masked": false
      }
    ]
  },
  {
    "token": 62,
    "prob": 0.9997590184211731,
    "text": "_",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 62,
        "prob": 0.9997590184211731,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 13,
        "prob": 0.000022090805941843428,
        "text": ".",
        "is_masked": false
      },
      {
        "token": 59,
        "prob": 0.000015420928320963867,
        "text": "\\",
        "is_masked": false
      },
      {
        "token": 834,
        "prob": 0.000013038138604315463,
        "text": "__",
        "is_masked": false
      },
      {
        "token": 12,
        "prob": 0.000011890577297890559,
        "text": "-",
        "is_masked": false
      }
    ]
  },
  {
    "token": 437,
    "prob": 0.7618218064308167,
    "text": "end",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 437,
        "prob": 0.7618218064308167,
        "text": "end",
        "is_masked": false
      },
      {
        "token": 9688,
        "prob": 0.17939522862434387,
        "text": "start",
        "is_masked": false
      },
      {
        "token": 27471,
        "prob": 0.00941860768944025,
        "text": "begin",
        "is_masked": false
      },
      {
        "token": 11338,
        "prob": 0.005442698486149311,
        "text": "stop",
        "is_masked": false
      },
      {
        "token": 1571,
        "prob": 0.0011666559148579836,
        "text": "ending",
        "is_masked": false
      }
    ]
  },
  {
    "token": 91,
    "prob": 0.9977994561195374,
    "text": "|",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 91,
        "prob": 0.9977994561195374,
        "text": "|",
        "is_masked": false
      },
      {
        "token": 62,
        "prob": 0.00038299127481877804,
        "text": "_",
        "is_masked": false
      },
      {
        "token": 15886,
        "prob": 0.00029917186475358903,
        "text": "||",
        "is_masked": false
      },
      {
        "token": 29,
        "prob": 0.0001353792322333902,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 10,
        "prob": 0.00012695405166596174,
        "text": "+",
        "is_masked": false
      }
    ]
  },
  {
    "token": 29,
    "prob": 0.9984844326972961,
    "text": ">",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 29,
        "prob": 0.9984844326972961,
        "text": ">",
        "is_masked": false
      },
      {
        "token": 60,
        "prob": 0.00033915083622559905,
        "text": "]",
        "is_masked": false
      },
      {
        "token": 6927,
        "prob": 0.00010037984611699358,
        "text": "><",
        "is_masked": false
      },
      {
        "token": 28,
        "prob": 0.00008626521594123915,
        "text": "=",
        "is_masked": false
      },
      {
        "token": 24618,
        "prob": 0.00007595126226078719,
        "text": ">\"",
        "is_masked": false
      }
    ]
  },
  {
    "token": 198,
    "prob": 0.5057048797607422,
    "text": "\n",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 198,
        "prob": 0.5057048797607422,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 1640,
        "prob": 0.21033842861652374,
        "text": "for",
        "is_masked": false
      },
      {
        "token": 10057,
        "prob": 0.032438330352306366,
        "text": "system",
        "is_masked": false
      },
      {
        "token": 7220,
        "prob": 0.028024563565850258,
        "text": "user",
        "is_masked": false
      },
      {
        "token": 329,
        "prob": 0.010914873331785202,
        "text": " for",
        "is_masked": false
      }
    ]
  },
  {
    "token": 31175,
    "prob": 0.0000045710116864938755,
    "text": ">:",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 27,
        "prob": 0.3989731967449188,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 1639,
        "prob": 0.1570923775434494,
        "text": "You",
        "is_masked": false
      },
      {
        "token": 49,
        "prob": 0.03772936016321182,
        "text": "R",
        "is_masked": false
      },
      {
        "token": 464,
        "prob": 0.027823705226182938,
        "text": "The",
        "is_masked": false
      },
      {
        "token": 3556,
        "prob": 0.01715909130871296,
        "text": "</",
        "is_masked": false
      },
      {
        "token": 31175,
        "prob": 0.0000045710116864938755,
        "text": ">:",
        "is_masked": false
      }
    ]
  },
  {
    "token": 7,
    "prob": 0.0029301203321665525,
    "text": "(",
    "is_masked": false,
    "latency_ms": 0,
    "is_generated": false,
    "is_force_forwarded": false,
    "is_input": true,
    "top_k": [
      {
        "token": 16,
        "prob": 0.0514117032289505,
        "text": "1",
        "is_masked": false
      },
      {
        "token": 27,
        "prob": 0.03453383222222328,
        "text": "<",
        "is_masked": false
      },
      {
        "token": 15,
        "prob": 0.03179513290524483,
        "text": "0",
        "is_masked": false
      },
      {
        "token": 352,
        "prob": 0.01785029098391533,
        "text": " 1",
        "is_masked": false
      },
      {
        "token": 198,
        "prob": 0.0139352697879076,
        "text": "\n",
        "is_masked": false
      },
      {
        "token": 7,
        "prob": 0.0029301203321665525,
        "text": "(",
        "is_masked": false
      }
    ]
  }
];