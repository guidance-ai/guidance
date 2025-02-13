// Metrics and their definitions.

import type { MetricDef } from './interfaces';

export const metricDefs: Record<string, MetricDef> = {
  'status': {
    name: 'Status',
    units: '',
    description: 'Determines whether engine is running, completed or in error.',
    isScalar: true,
    precision: 0
  },
  'cpu': {
    name: 'CPU',
    units: '%',
    description: 'Average utilization across CPU cores.',
    isScalar: false,
    precision: 1
  },
  'gpu': {
    name: 'GPU',
    units: '%',
    description: 'Average utilization across GPUs.',
    isScalar: false,
    precision: 1
  },
  'ram': {
    name: 'RAM',
    units: 'GB',
    description: 'Utilization of RAM.',
    isScalar: true,
    precision: 1
  },
  'vram': {
    name: 'VRAM',
    units: 'GB',
    description: 'Utilization of video RAM.',
    isScalar: true,
    precision: 1
  },
  'wall time': {
    name: 'Time',
    units: 's',
    description: 'Time taken from initial display to engine completion.',
    isScalar: true,
    precision: 1
  },
  'avg latency': {
    name: 'Avg Latency',
    units: 'ms',
    description: 'Average roundtrip latency per token',
    isScalar: true,
    precision: 0
  },
  'consumed': {
    name: 'Consumed',
    units: 'tkn',
    description: 'Total tokens consumed by language model.',
    isScalar: true,
    precision: 0
  },
  'token reduction': {
    name: 'Token Reduction',
    units: '%',
    description: 'Total tokens consumed by language model divided by total tokens.',
    isScalar: true,
    precision: 0
  }
};
