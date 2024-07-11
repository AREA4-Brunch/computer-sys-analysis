import abc
from collections import defaultdict
from typing import Callable


class IObservable(abc.ABC):
    @abc.abstractmethod
    def subscribe(self, event: any, notify_strategy: Callable):
        pass

    @abc.abstractmethod
    def unsubscribe(self, event: any, notify_strategy: Callable):
        pass

    @abc.abstractmethod
    def notify(self, event: any, *args, **kwargs):
        pass

    @abc.abstractmethod
    def num_subscribers(self, event: any) -> int:
        pass


class Observable(IObservable):
    """ Observable can be used via inheritance or composition.
        Does not impose any interface on notify strategy of a
        subscriber and just passes given args in notify.
        Groups subscribers by given types of events.
    """
    # _subscribers: dict, subscribers[event] = [ notify_strategies... ]

    def __init__(self):
        self._subscribers = defaultdict(lambda: [])

    def subscribe(self, event: any, notify_strategy: Callable):
        self._subscribers[event].append(notify_strategy)

    def unsubscribe(self, event: any, notify_strategy: Callable):
        self._subscribers[event].remove(notify_strategy)

    def notify(self, event: any, *args, **kwargs):
        for notify_strategy in self._subscribers[event]:
            notify_strategy(*args, **kwargs)

    def num_subscribers(self, event: any) -> int:
        return len(self._subscribers[event])
