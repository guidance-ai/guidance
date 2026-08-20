import sys
from types import SimpleNamespace

import pytest

from guidance.metrics import _metrics
from guidance.registry import get_bg_async


def test_monitor_limits_metric_history(monkeypatch):
    monitor = _metrics.Monitor(interval_ms=0, max_size=2)

    class GPUStats:
        def __init__(self, gpus):
            self.gpus = gpus

        def __len__(self):
            return len(self.gpus)

    cpu_samples = iter([[10.0], [20.0], [30.0]])
    memory_samples = iter([1, 2, 3])
    gpu_queries = iter(
        [
            GPUStats([SimpleNamespace()]),
            GPUStats([SimpleNamespace(utilization=10.0, memory_used=100, memory_total=1000)]),
            GPUStats([SimpleNamespace(utilization=20.0, memory_used=200, memory_total=1000)]),
            GPUStats([SimpleNamespace(utilization=30.0, memory_used=300, memory_total=1000)]),
        ]
    )
    fake_gpustat = SimpleNamespace(GPUStatCollection=SimpleNamespace(new_query=lambda: next(gpu_queries)))

    monkeypatch.setitem(sys.modules, "gpustat", fake_gpustat)
    monkeypatch.setattr(
        _metrics.psutil,
        "cpu_percent",
        lambda *, interval, percpu: next(cpu_samples),
    )

    def virtual_memory():
        used_gb = next(memory_samples)
        if used_gb == 3:
            monitor.stop_flag = True
        return SimpleNamespace(used=used_gb * 1024**3)

    monkeypatch.setattr(
        _metrics.psutil,
        "virtual_memory",
        virtual_memory,
    )

    get_bg_async().run_async_coroutine(monitor._monitor_fn()).result()

    assert monitor.metrics_dict == {
        _metrics.MonitoringMetric.CPU_USAGE: pytest.approx([0.2, 0.3]),
        _metrics.MonitoringMetric.MEM_USAGE: pytest.approx([2.0, 3.0]),
        _metrics.MonitoringMetric.GPU_USAGE: [[0.2], [0.3]],
        _metrics.MonitoringMetric.GPU_USED_MEM: [[200], [300]],
        _metrics.MonitoringMetric.GPU_TOTAL_MEM: [[1000], [1000]],
    }
