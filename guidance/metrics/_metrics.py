import asyncio
import logging
import time
from asyncio import CancelledError
from enum import Enum
from typing import Any, Sequence

import psutil

from .._topics import METRICS_TOPIC
from .._types import TokenUsage
from ..visual import MetricMessage

MISSING_VALUE = 0

logger = logging.getLogger(__name__)


class PeriodicMetricsGenerator:
    def __init__(self, monitor: "Monitor", sleep_sec=0.5):
        self._monitor = monitor
        self._sleep_sec = sleep_sec
        self._task = None
        self._cancelled = False
        self._is_paused = False

    def start(self):
        from ..registry import get_bg_async

        bg = get_bg_async()
        self._task = bg.run_async_coroutine(bg.async_task(self._emit())).result()

    def stop(self):
        if self._task is not None:
            self._cancelled = True
            self._task.cancel()

    def pause(self):
        """
        Pauses the model by setting the internal _is_paused flag to True.

        This method can be used to temporarily halt the model's operations.
        """
        self._is_paused = True

    def resume(self):
        """
        Resume the model's operation by setting the paused state to False.

        This method changes the internal state of the model to indicate that it is no longer paused.
        """
        self._is_paused = False

    async def _emit(self):
        import asyncio

        from ..registry import get_exchange

        while not self._cancelled:
            try:
                await asyncio.sleep(self._sleep_sec)

                cpu_percent = self._monitor.get_metric(MonitoringMetric.CPU_USAGE)
                used_ram = self._monitor.get_metric(MonitoringMetric.MEM_USAGE)
                gpu_percent = self._monitor.get_metric(MonitoringMetric.GPU_USAGE)
                gpu_used_vram = self._monitor.get_metric(MonitoringMetric.GPU_USED_MEM)

                if gpu_percent:
                    gpu_percent = max(gpu_percent)
                else:
                    gpu_percent = MISSING_VALUE

                if gpu_used_vram:
                    gpu_used_vram = max(gpu_used_vram)
                else:
                    gpu_used_vram = MISSING_VALUE

                if not cpu_percent:
                    cpu_percent = MISSING_VALUE

                if not used_ram:
                    used_ram = MISSING_VALUE

                if not self._is_paused:
                    exchange = get_exchange()
                    exchange.publish(MetricMessage(name="cpu", value=cpu_percent), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="ram", value=used_ram), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="gpu", value=gpu_percent), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="vram", value=gpu_used_vram), topic=METRICS_TOPIC)
            except CancelledError:
                logger.debug("METRICGEN:canceling")
                break
            except Exception as e:  # noqa BLE001
                logger.debug(f"METRICGEN: {repr(e)}", exc_info=True)
                break

        logger.debug("METRICGEN:exiting")


class MonitoringMetric(str, Enum):
    CPU_USAGE = "cpu_usage"
    MEM_USAGE = "mem_usage"
    GPU_USAGE = "gpu_usage"
    GPU_USED_MEM = "gpu_used_mem"
    GPU_TOTAL_MEM = "gpu_total_mem"


class Monitor:
    """Monitoring service to collect necessary metrics for visualization"""

    def __init__(self, interval_ms: int = 1000, **kwargs):
        self.max_size = kwargs.get("max_size", 100)
        self.stop_flag = False
        self.task = None
        self.interval_ms = interval_ms

        # Initialize metrics storage
        self.metrics_dict = {
            MonitoringMetric.CPU_USAGE: [],
            MonitoringMetric.MEM_USAGE: [],
            MonitoringMetric.GPU_USAGE: [],
            MonitoringMetric.GPU_USED_MEM: [],
            MonitoringMetric.GPU_TOTAL_MEM: [],
        }

    async def _monitor_fn(self):
        to_collect_gpu_stats = False
        has_gpustat = False
        try:
            import gpustat

            has_gpustat = True
        except ImportError:
            logger.warning("gpustat is not installed, run `pip install gpustat` to collect GPU stats.")

        if has_gpustat:
            try:
                gpu_stats = gpustat.GPUStatCollection.new_query()
                if len(gpu_stats) > 0:
                    to_collect_gpu_stats = True
            except Exception as e:  # noqa BLE001
                logger.warning(f"Non-Nvidia GPU monitoring is not supported in this version. {e}", exc_info=True)

        while not self.stop_flag:
            try:
                t0 = time.time()

                cpu_per_core_percent = psutil.cpu_percent(interval=None, percpu=True)
                average_cpu_percent = sum(cpu_per_core_percent) / len(cpu_per_core_percent)
                average_cpu_utilization = average_cpu_percent / 100.0
                memory_usage = psutil.virtual_memory()
                memory_usage_gb = memory_usage.used / (1024**3)

                self.metrics_dict[MonitoringMetric.CPU_USAGE].append(average_cpu_utilization)
                self.metrics_dict[MonitoringMetric.MEM_USAGE].append(memory_usage_gb)
                if to_collect_gpu_stats:
                    gpu_stats = gpustat.GPUStatCollection.new_query()

                    usage = [gpu.utilization / 100.0 for gpu in gpu_stats.gpus]
                    mem_usage = [gpu.memory_used for gpu in gpu_stats.gpus]
                    mem_total = [gpu.memory_total for gpu in gpu_stats.gpus]

                    self.metrics_dict[MonitoringMetric.GPU_USAGE].append(usage)
                    self.metrics_dict[MonitoringMetric.GPU_USED_MEM].append(mem_usage)
                    self.metrics_dict[MonitoringMetric.GPU_TOTAL_MEM].append(mem_total)

                # Trim lists to max_size
                for metrics in self.metrics_dict.values():
                    if len(metrics) > self.max_size:
                        metrics = metrics[-self.max_size :]

                lat = time.time() - t0
                sleep_time = self.interval_ms / 1000.0 - lat
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Caught {e}", exc_info=True)
                await asyncio.sleep(1)  # Wait a bit before retrying on error

    def start(self):
        from ..registry import get_bg_async

        bg = get_bg_async()
        self.stop_flag = False
        self.task = bg.run_async_coroutine(bg.async_task(self._monitor_fn())).result()
        logger.debug("MONITOR:start")

    def stop(self):
        self.stop_flag = True
        if self.task:
            self.task.cancel()
            self.task = None
        for metrics in self.metrics_dict.values():
            metrics.clear()
        logger.debug("MONITOR:stop")

    def reset(self):
        self.stop()
        self.start()
        logger.debug("MONITOR:reset")

    def get_metrics(
        self,
        metrics: Sequence[MonitoringMetric] = (),
    ) -> dict[MonitoringMetric, Any]:
        if not metrics:
            metrics = MonitoringMetric.__members__.values()
        result = {}
        for metric in metrics:
            if metric in MonitoringMetric.__members__.values():
                result[metric] = self.metrics_dict[metric][-1] if len(self.metrics_dict[metric]) > 0 else None
            else:
                raise ValueError(f"Unknown monitoring metric: {metric}")
        return result

    def get_metric(self, metric: MonitoringMetric) -> Any:
        return self.get_metrics([metric])[metric]


def emit_usage(usage: TokenUsage) -> None:
    from ..registry import get_exchange

    exchange = get_exchange()

    exchange.publish(
        MetricMessage(
            name="token reduction",
            value=(usage.token_savings or 0) * 100,  # display as percentage
        ),
        topic=METRICS_TOPIC,
    )

    exchange.publish(MetricMessage(name="consumed", value=usage.forward_passes), topic=METRICS_TOPIC)

    exchange.publish(MetricMessage(name="avg latency", value=usage.avg_latency_ms), topic=METRICS_TOPIC)
