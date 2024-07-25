import abc
import concurrent.futures
import threading
from typing import Callable
from ..utils.observable import IObservable, Observable
from ..core.timer import ITimer
from .timer import Timer
from ..core.scheduler import ITasksScheduler
from .scheduler import TasksScheduler


class ISimulationObservable(abc.ABC):
    @abc.abstractmethod
    def subscribe(self, event: any, notify_strategy: Callable) -> Callable:
        pass

    @abc.abstractmethod
    def unsubscribe(self, event: any, notify_strategy: Callable):
        pass


class ISimulation(ISimulationObservable):
    class Event:
        ON_START = 1  # push (sim: ISimulation)
        ON_END   = 2  # push (sim: ISimulation)

    @abc.abstractmethod
    def simulate(self, duration: float):
        pass

    @abc.abstractmethod
    def timer(self) -> ITimer:
        pass

    @abc.abstractmethod
    def scheduler(self) -> ITasksScheduler:
        pass

    @abc.abstractmethod
    def duration(self) -> float:
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
        self._duration = 0

    def subscribe(self, event: str, notify_strategy: Callable):
        self.__observable.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable):
        self.__observable.unsubscribe(event, notify_strategy)

    def simulate(self, max_duration: float):
        """ Starts the simulation that lasts until the time in
            the simulation is >= `max_duration`.
            Triggers Event.ON_START and Event.ON_END.
        """
        self._timer = Timer(start_time=0)
        self._scheduler = TasksScheduler(self._timer)
        self._on_sim_start()
        while self._scheduler.has_next():
            scheduled_at = self._scheduler.next_scheduled_at()
            if scheduled_at >= max_duration: break;
            self._timer.set_now(scheduled_at)
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
        self._duration = self._timer.now()
        self._notify(Simulation.Event.ON_END, self)

    def duration(self) -> float:
        return self._duration


class SequentialSimulations(ISimulation):
    """ Composite in composite design pattern. """
    # _sims: list[ISimulation]

    def __init__(self) -> None:
        super().__init__()
        self._sims: list[ISimulation] = []

    def add(self, simulation: ISimulation) -> 'SequentialSimulations':
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

    def duration(self) -> float:
        duration = 0
        for sim in self._sims:
            duration += sim.duration()
        return duration


class MultiThreadedSimulations(ISimulation):
    """ Composite in composite design pattern.
        Uses `threading` module, does not utilize CPU's multiple cores.
    """
    # _sims: list[ISimulation]

    def __init__(self, max_workers: int | None=None) -> None:
        super().__init__()
        self._sims: list[ISimulation] = []
        self._max_workers = max_workers

    def add(self, simulation: ISimulation) -> 'MultiThreadedSimulations':
        self._sims.append(simulation)
        return self

    def subscribe(self, event: str, notify_strategy: Callable):
        for sim in self._sims:
            sim.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable):
        for sim in self._sims:
            sim.unsubscribe(event, notify_strategy)

    def simulate(self, duration: float):
        threads = []
        for sim in self._sims:
            thread = threading.Thread(target=sim.simulate, args=(duration,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def timer(self) -> ITimer:
        return None

    def scheduler(self) -> ITasksScheduler:
        return None

    def duration(self) -> float:
        duration = 0
        for sim in self._sims:
            duration += sim.duration()
        return duration


class MultiProcessedSimulations(ISimulation):
    """ Composite in composite design pattern.
        Uses `concurrent` module, utilizes CPU's multiple cores.
    """
    # _sims: list[ISimulation]

    def __init__(self, max_workers: int | None=None) -> None:
        super().__init__()
        self._sims: list[ISimulation] = []
        self._max_workers = max_workers

    def add(self, simulation: ISimulation) -> 'MultiProcessedSimulations':
        self._sims.append(simulation)
        return self

    def subscribe(self, event: str, notify_strategy: Callable):
        for sim in self._sims:
            sim.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable):
        for sim in self._sims:
            sim.unsubscribe(event, notify_strategy)

    def simulate(self, duration: float):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers,
        ) as executor:
            futures = [
                executor.submit(sim.simulate, duration)
                for sim in self._sims
            ]
            concurrent.futures.as_completed(futures)

    def timer(self) -> ITimer:
        return None

    def scheduler(self) -> ITasksScheduler:
        return None

    def duration(self) -> float:
        duration = 0
        for sim in self._sims:
            duration += sim.duration()
        return duration
