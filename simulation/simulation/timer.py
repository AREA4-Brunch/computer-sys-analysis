from ..core.timer import ITimer


class Timer(ITimer):
    # _now: float - current time

    def __init__(self, start_time: float=0):
        super().__init__()
        self._now = start_time

    def now(self) -> float:
        return self._now

    def set_now(self, new_time: float):
        self._now = new_time
