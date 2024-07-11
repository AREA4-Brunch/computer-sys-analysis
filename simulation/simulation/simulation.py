import abc
from typing import Callable
from utils.observable import IObservable, Observable
from simulation.utils.timer import ITimer, Timer
from simulation.scheduler import ITasksScheduler, TasksScheduler


class ISimulationObservable(abc.ABC):
    @abc.abstractmethod
    def subscribe(self, event: any, notify_strategy: Callable) -> Callable:
        pass

    @abc.abstractmethod
    def unsubscribe(self, event: any, notify_strategy: Callable):
        pass


class ISimulation(ISimulationObservable):
    class Event:
        ON_START = 1
        ON_END   = 2

    @abc.abstractmethod
    def simulate(self, duration: float):
        pass

    @abc.abstractmethod
    def timer(self) -> ITimer:
        pass

    @abc.abstractmethod
    def scheduler(self) -> ITasksScheduler:
        pass


class Simulation(ISimulation):
    """ Leaf node in composite design pattern.
        Pull configuration of the observer design pattern.
    """
    # __observable: IObservable
    # _timer: ITimer | None
    # _scheduler: ITasksScheduler | None

    def __init__(self):
        super().__init__()
        self.__observable: IObservable = Observable()
        self._timer = None
        self._scheduler = None

    def subscribe(self, event: str, notify_strategy: Callable):
        self.__observable.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable):
        self.__observable.unsubscribe(event, notify_strategy)

    def simulate(self, duration: float):
        """ Starts the simulation that lasts until the time in
            the simulation is >= `duration`.
            Triggers Event.ON_START and Event.ON_END.
        """
        self._timer = Timer(start_time=0)
        self._scheduler = TasksScheduler(self._timer)
        self._on_sim_start()
        while self._scheduler.has_next():
            scheduled_at = self._scheduler.next_scheduled_at()
            if scheduled_at >= duration: break;
            self._timer.now(scheduled_at)
            func, args, kwargs = self._scheduler.next()
            func(*args, **kwargs)
        self._on_sim_end()

    def timer(self) -> ITimer:
        """ Interface ret type since it provides only expected control
            over timer.
        """
        return self._timer

    def scheduler(self) -> ITasksScheduler:
        """ Interface ret type since it provides only expected control
            over scheduler.
        """
        return self._scheduler

    def _notify(self, event: str, *args, **kwargs):
        self.__observable.notify(event, *args, **kwargs)

    def _on_sim_start(self):
        self._notify(Simulation.Event.ON_START, self)

    def _on_sim_end(self):
        self._notify(Simulation.Event.ON_END, self)


class ParallelSimulations(ISimulation):
    """ Composite in composite design pattern. """
    # _sims: list[ISimulation]

    def __init__(self) -> None:
        super().__init__()
        self._sims: list[ISimulation] = []

    def add(self, simulation: ISimulation) -> 'ParallelSimulations':
        self._sims.append(simulation)
        return self

    def subscribe(self, event: str, notify_strategy: Callable):
        for sim in self._sims:
            sim.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable):
        for sim in self._sims:
            sim.unsubscribe(event, notify_strategy)

    def simulate(self, duration: float):
        for sim in self._sims:
            sim.simulate(duration)

    def timer(self) -> ITimer:
        return None

    def scheduler(self) -> ITasksScheduler:
        return None
