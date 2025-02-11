// Metrics and their definitions.

import type { MetricDef } from './interfaces';

export const metricDefs: Record<string, MetricDef> = {
  'status': {
    name: 'status',
    units: '',
    description: 'Determines whether engine is running, completed or in error.',
    isScalar: true,
    precision: 0
  },
  'cpu': {
    name: 'cpu',
    units: '%',
    description: 'Average utilization across CPU cores.',
    isScalar: false,
    precision: 1
  },
  'gpu': {
    name: 'gpu',
    units: '%',
    description: 'Average utilization across GPUs.',
    isScalar: false,
    precision: 1
  },
  'ram': {
    name: 'ram',
    units: 'GB',
    description: 'Utilization of RAM.',
    isScalar: true,
    precision: 1
  },
  'vram': {
    name: 'vram',
    units: 'GB',
    description: 'Utilization of video RAM.',
    isScalar: true,
    precision: 1
  },
  'wall time': {
    name: 'wall time',
    units: 's',
    description: 'Time taken from initial display to engine completion.',
    isScalar: true,
    precision: 1
  },
  'avg latency': {
    name: 'avg latency',
    units: 'ms',
    description: 'Average roundtrip latency per token',
    isScalar: true,
    precision: 0
  },
  'consumed': {
    name: 'consumed',
    units: 'tkn',
    description: 'Total tokens consumed by language model.',
    isScalar: true,
    precision: 0
  },
  'token reduction': {
    name: 'token reduction',
    units: '%',
    description: 'Total tokens consumed by language model divided by total tokens.',
    isScalar: true,
    precision: 0
  }
};
