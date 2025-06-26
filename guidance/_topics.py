"""Exchange topic constants for guidance message routing.

This module centralizes all topic constants used throughout the guidance
codebase for the TopicExchange message routing system.
"""

# Default topic for general message routing
DEFAULT_TOPIC = "/default"

# Metrics-related topics
METRICS_TOPIC = "/metrics"

# Trace topics
TRACE_TOPIC = "/trace"

__all__ = [
    "DEFAULT_TOPIC",
    "METRICS_TOPIC", 
    "TRACE_TOPIC",
]