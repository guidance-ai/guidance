""" Poor man's exchanges for routing messages. """

from collections import defaultdict
from typing import Callable
from ..visual import GuidanceMessage
import re
import logging

logger = logging.getLogger(__name__)

DEFAULT_TOPIC = "/default"
WILDCARD_PATTERN = r".*"


class TopicExchange:
    """ Queue-less topic exchange for routing messages.

    This is not as comprehensive as a full distributed topic exchange.
    It is specific to a single process, with no queues and less generalized routing keys.
    """

    def __init__(self):
        """ Initializes."""
        self._observers = defaultdict(list)

    def subscribe(self, callback: Callable[[GuidanceMessage], None], topic_pat: str = WILDCARD_PATTERN) -> None:
        """ Subscribes to incoming messages.

        Args:
            callback: Callback to handle incoming messages.
            topic_pat: Topic to notify.
        """
        logger.debug(f"EXCHANGE:pre_subscribe:{self._observers[topic_pat]}")
        self._observers[topic_pat].append(callback)
        logger.debug(f"EXCHANGE:post_subscribe:{self._observers[topic_pat]}")

    def unsubscribe(self, callback: Callable[[GuidanceMessage], None], topic_pat: str = WILDCARD_PATTERN) -> None:
        """ Unsubscribes from incoming messages.

        Args:
            callback: Callback to remove.
            topic_pat: Topic pattern.
        """
        logger.debug(f"EXCHANGE:pre_unsubscribe:{self._observers[topic_pat]}")
        try:
            self._observers[topic_pat].remove(callback)
        except ValueError as _:
            logger.warning(f"EXCHANGE:cb at '{topic_pat}' already removed.")
        logger.debug(f"EXCHANGE:post_unsubscribe:{self._observers[topic_pat]}")

        if len(self._observers[topic_pat]) == 0:
            logger.debug(f"EXCHANGE:delete_entry:{topic_pat}")
            del self._observers[topic_pat]

    def publish(self, message: GuidanceMessage, topic: str = DEFAULT_TOPIC):
        """ Notifies all subscribers to topic pattern of an incoming message.

        Args:
            message: Incoming message.
            topic: Topics to notify.
        """
        # logger.debug(f"EXCHANGE:publish:{message}")
        for obs_topic_pat, observers in self._observers.items():
            if re.match(obs_topic_pat, topic):
                for observer in observers:
                    observer(message)


__all__ = ["TopicExchange"]
