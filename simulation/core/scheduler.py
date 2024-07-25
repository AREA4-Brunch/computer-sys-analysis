import abc


class ITasksScheduler(abc.ABC):
    """ Provides the interface for registering into the
        scheduler, but not for manipulating it.
    """
    @abc.abstractmethod
    def add(self, schedule_in: float, func: callable, *args, **kwargs):
        pass
