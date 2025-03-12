import time
from asyncio import CancelledError
from enum import Enum
from multiprocessing import Manager, Process
from typing import Union, Any
import logging

import psutil

from .._schema import GuidanceEngineMetrics
from ..models import Model

from ..visual import MetricMessage

METRICS_TOPIC = "/metrics"
MISSING_VALUE = 0

logger = logging.getLogger(__name__)


class PeriodicMetricsGenerator:
    def __init__(self, monitor: "Monitor", sleep_sec=0.5):
        self._monitor = monitor
        self._sleep_sec = sleep_sec
        self._task = None
        self._task_cancelled = False
        self._is_paused = False

    def start(self):
        from ..registry import get_bg_async
        bg = get_bg_async()
        self._task = bg.run_async_coroutine(bg.async_task(self._emit())).result()

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task_cancelled = True

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
        import time
        from ..registry import get_exchange

        time_start = time.time()
        while not self._task_cancelled:
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

                time_end = time.time()
                time_elapsed = time_end - time_start

                if not self._is_paused:
                    exchange = get_exchange()
                    exchange.publish(MetricMessage(name="wall time", value=time_elapsed), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="cpu", value=cpu_percent), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="ram", value=used_ram), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="gpu", value=gpu_percent), topic=METRICS_TOPIC)
                    exchange.publish(MetricMessage(name="vram", value=gpu_used_vram), topic=METRICS_TOPIC)
            except CancelledError:
                logger.debug("METRICGEN:canceling")
                break
            except Exception as e:
                logger.debug(f"METRICGEN: {repr(e)}")
                break

        logger.debug("METRICGEN:exiting")


# TODO(nopdive): @hudson-ai needs refactor into engine/class?
class PostExecMetrics:
    def __init__(self, monitor: "Monitor"):
        self._monitor = monitor

    def emit_messages(self, lm: "Model"):
        from ..registry import get_exchange
        exchange = get_exchange()

        token_reduction = self._monitor.get_metric(MonitoringMetric.TOKEN_REDUCTION, lm)

        if token_reduction is not None:
            exchange.publish(MetricMessage(
                name="token reduction",
                value=token_reduction * 100,
            ), topic=METRICS_TOPIC)

        output_tokens = self._monitor.get_metric(MonitoringMetric.OUTPUT_TOKENS, lm)
        if output_tokens is not None:
            exchange.publish(MetricMessage(name="consumed", value=output_tokens), topic=METRICS_TOPIC)

        avg_latency = self._monitor.get_metric(MonitoringMetric.AVG_LATENCY, lm)
        if avg_latency is not None:
            exchange.publish(MetricMessage(name="avg latency", value=avg_latency), topic=METRICS_TOPIC)


class MonitoringMetric(str, Enum):
    CPU_USAGE = "cpu_usage"
    MEM_USAGE = "mem_usage"
    GPU_USAGE = "gpu_usage"
    GPU_USED_MEM = "gpu_used_mem"
    GPU_TOTAL_MEM = "gpu_total_mem"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    BACKTRACK_TOKENS = "backtrack_tokens"
    TOKEN_COUNT = "token_count"
    TOKEN_REDUCTION = "token_reduction"
    AVG_LATENCY = "avg_latency"


ALL_METRICS = [
    MonitoringMetric.CPU_USAGE,
    MonitoringMetric.MEM_USAGE,
    MonitoringMetric.GPU_USAGE,
    MonitoringMetric.GPU_USED_MEM,
    MonitoringMetric.GPU_TOTAL_MEM,
    MonitoringMetric.INPUT_TOKENS,
    MonitoringMetric.OUTPUT_TOKENS,
    MonitoringMetric.BACKTRACK_TOKENS,
    MonitoringMetric.TOKEN_COUNT,
    MonitoringMetric.TOKEN_REDUCTION,
    MonitoringMetric.AVG_LATENCY,
]


def _monitor_fn(
        stop_flag,
        metrics_dict: dict[MonitoringMetric, list],
        max_size: int = 100,
        interval_ms: float = 1000,
):
    # print("Monitoring started")

    to_collect_gpu_stats = False
    has_gpustat = False
    try:
        import gpustat

        has_gpustat = True
    except:
        logger.warning("gpustat is not installed, run `pip install gpustat` to collect GPU stats.")

    if has_gpustat:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            if len(gpu_stats) > 0:
                # only collect GPU stats if there is at least one GPU
                to_collect_gpu_stats = True
        except:
            logger.warning("Non-Nvidia GPU monitoring is not supported in this version.")

    p = psutil.Process()
    try:
        while not stop_flag.value:
            t0 = time.time()

            cpu_percent = p.cpu_percent() / 100.0
            memory_usage = psutil.virtual_memory()

            metrics_dict[MonitoringMetric.CPU_USAGE].append(cpu_percent)
            metrics_dict[MonitoringMetric.MEM_USAGE].append(memory_usage.used / (1024**3))

            if to_collect_gpu_stats:
                gpu_stats = gpustat.GPUStatCollection.new_query()

                usage = [gpu.utilization / 100.0 for gpu in gpu_stats.gpus]
                mem_usage = [gpu.memory_used for gpu in gpu_stats.gpus]
                mem_total = [gpu.memory_total for gpu in gpu_stats.gpus]

                metrics_dict[MonitoringMetric.GPU_USAGE].append(usage)
                metrics_dict[MonitoringMetric.GPU_USED_MEM].append(mem_usage)
                metrics_dict[MonitoringMetric.GPU_TOTAL_MEM].append(mem_total)

            for metrics in metrics_dict.values():
                if len(metrics) > max_size:
                    metrics.pop(0)

            lat = time.time() - t0

            # sleep for the remaining time of the interval
            sleep_time = interval_ms / 1000.0 - lat
            if sleep_time < 0:
                time.sleep(sleep_time)
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")
        pass

    # print("Monitoring stopped")


class Monitor:
    """Monitoring service to collect necessary metrics for visualization"""

    def __init__(self, engine_metrics: GuidanceEngineMetrics, **kwargs):
        self.engine_metrics = engine_metrics
        self.mp_manager = Manager()

        # use list instead of queue for easily accessing each item, e.g., last item
        self.max_size = kwargs.get("max_size", 100)

        self.metrics_dict = {
            MonitoringMetric.CPU_USAGE: self.mp_manager.list(),
            MonitoringMetric.MEM_USAGE: self.mp_manager.list(),
            MonitoringMetric.GPU_USAGE: self.mp_manager.list(),
            MonitoringMetric.GPU_USED_MEM: self.mp_manager.list(),
            MonitoringMetric.GPU_TOTAL_MEM: self.mp_manager.list(),
        }

        self.stop_flag = self.mp_manager.Value("b", False)
        self.process = None

        self.per_token_metrics = []  # store metrics per token in token list

    def start(self):
        self.process = Process(
            target=_monitor_fn, args=(self.stop_flag, self.metrics_dict, self.max_size)
        )
        self.process.start()
        logger.debug("MONITOR:start")

    def stop(self):
        if self.process:
            self.stop_flag.value = True
            self.process.terminate()

            for metrics in self.metrics_dict.values():
                metrics[:] = []  # NOTE(nopdive): ListProxy does not have .clear method.
        logger.debug("MONITOR:stop")

    def reset(self):
        self.stop()

        for metrics in self.metrics_dict.values():
            metrics.clear()

        self.start()
        logger.debug("MONITOR:reset")

    def get_metrics(
            self, metrics=None, lm: Union["Model", None] = None
    ) -> dict[MonitoringMetric, Any]:
        if metrics is None:
            metrics = ALL_METRICS
        result = {}

        for metric in metrics:
            if metric in [
                MonitoringMetric.CPU_USAGE,
                MonitoringMetric.MEM_USAGE,
                MonitoringMetric.GPU_USAGE,
                MonitoringMetric.GPU_USED_MEM,
                MonitoringMetric.GPU_TOTAL_MEM,
            ]:
                result[metric] = (
                    self.metrics_dict[metric][-1] if len(self.metrics_dict[metric]) > 0 else None
                )
            elif metric == MonitoringMetric.INPUT_TOKENS:
                result[metric] = self.engine_metrics.engine_input_tokens
            elif metric == MonitoringMetric.OUTPUT_TOKENS:
                result[metric] = self.engine_metrics.engine_output_tokens
            elif metric == MonitoringMetric.BACKTRACK_TOKENS:
                result[metric] = self.engine_metrics.engine_backtrack_tokens
            elif metric == MonitoringMetric.TOKEN_COUNT:
                result[metric] = lm.token_count if lm is not None else None
            elif metric == MonitoringMetric.TOKEN_REDUCTION:
                if lm is not None and lm.token_count > 0:
                    result[metric] = 1 - min(1, (lm.metrics.engine_output_tokens / lm.token_count))
                else:
                    result[metric] = None
            elif metric == MonitoringMetric.AVG_LATENCY:
                # TODO(nopdive): Fix this for IR refactor.
                result[metric] = None
        return result

    def get_metric(self, metric: MonitoringMetric, lm: Union["Model", None] = None) -> Any:
        return self.get_metrics([metric], lm)[metric]
