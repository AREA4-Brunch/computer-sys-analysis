import abc


class ITimer(abc.ABC):
    """ Does not provide change of time functionality. """

    @abc.abstractmethod
    def now(self) -> float:
        pass


class Timer(ITimer):
    """ Provides change of time functionality. """
    # _now: float - current time

    def __init__(self, start_time: float=0):
        super().__init__()
        self._now = start_time

    def now(self) -> float:
        return self._now

    def now(self, new_time: float):
        self._now = new_time
