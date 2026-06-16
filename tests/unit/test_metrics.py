from guidance.metrics._metrics import Monitor, MonitoringMetric


def test_trim_metrics_caps_each_list_to_max_size():
    monitor = Monitor(max_size=3)

    cpu = MonitoringMetric.CPU_USAGE
    mem = MonitoringMetric.MEM_USAGE
    monitor.metrics_dict[cpu].extend([0.1, 0.2, 0.3, 0.4, 0.5])
    monitor.metrics_dict[mem].extend([1.0, 2.0])

    monitor._trim_metrics()

    # Over-long lists are trimmed to the most-recent max_size entries...
    assert monitor.metrics_dict[cpu] == [0.3, 0.4, 0.5]
    # ...while short lists are left untouched.
    assert monitor.metrics_dict[mem] == [1.0, 2.0]


def test_trim_metrics_mutates_stored_lists_in_place():
    monitor = Monitor(max_size=2)
    cpu = MonitoringMetric.CPU_USAGE
    stored = monitor.metrics_dict[cpu]
    stored.extend([1, 2, 3, 4])

    monitor._trim_metrics()

    # The same list object referenced by metrics_dict must be the trimmed one.
    assert monitor.metrics_dict[cpu] is stored
    assert monitor.metrics_dict[cpu] == [3, 4]
