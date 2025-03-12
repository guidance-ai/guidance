""" Poor man's exchanges for routing messages. """

from collections import defaultdict
from typing import Callable
from ..visual import GuidanceMessage
from fnmatch import fnmatch
import logging

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "/default"
WILDCARD_PATTERN = "*"

class TopicExchange:
    """ Queue-less topic exchange for routing messages.

    This is not as comprehensive as a full distributed topic exchange.
    It is specific to a single process, with no queues and less generalized routing keys.
    """

    def __init__(self):
        """ Initializes."""
        self._observers = defaultdict(list)

    def subscribe(self, callback: Callable[[GuidanceMessage], None], topic: str = DEFAULT_TOPIC) -> None:
        """ Subscribes to incoming messages.

        Args:
            callback: Callback to handle incoming messages.
            topic: Topic to notify.
        """
        logger.debug(f"EXCHANGE:pre_subscribe:{self._observers[topic]}")
        self._observers[topic].append(callback)
        logger.debug(f"EXCHANGE:post_subscribe:{self._observers[topic]}")

    def unsubscribe(self, callback: Callable[[GuidanceMessage], None], topic: str = DEFAULT_TOPIC) -> None:
        """ Unsubscribes from incoming messages.

        Args:
            callback: Callback to remove.
            topic: Topic to notify.
        """
        logger.debug(f"EXCHANGE:pre_unsubscribe:{self._observers[topic]}")
        try:
            self._observers[topic].remove(callback)
        except ValueError as _:
            logger.warning(f"EXCHANGE:cb at '{topic}' already removed.")
        logger.debug(f"EXCHANGE:post_unsubscribe:{self._observers[topic]}")

    def notify(self, message: GuidanceMessage, topic_pattern: str = WILDCARD_PATTERN):
        """ Notifies all subscribers to topic pattern of an incoming message.

        Args:
            message: Incoming message.
            topic_pattern: Topics to notify (uses fnmatch).
        """
        for observer_topic, observers in self._observers.items():
            if fnmatch(observer_topic, topic_pattern):
                for observer in observers:
                    observer(message)

__all__ = ["TopicExchange"]
