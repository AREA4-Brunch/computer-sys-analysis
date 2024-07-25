import abc


class ITimer(abc.ABC):
    """ Provides the interface for getting current time,
        but not for manipulating it.
    """

    @abc.abstractmethod
    def now(self) -> float:
        pass
