import abc


class ISimulatedResourceMetrics(abc.ABC):
    @abc.abstractmethod
    def add_processing_jobs_cnt(self, count: int):
        pass

    @abc.abstractmethod
    def add_processing_time_of_jobs(self, time: float):
        pass

    @abc.abstractmethod
    def add_jobs_cnt_during_time(self, count_time: float):
        """ Takes in number of jobs on a resource multiplied
            by time during which that number has not changed.
        """
        pass

    @abc.abstractmethod
    def set_total_time_passed(self, duration: float):
        """ Set total time over which metrics were being gathered/added.
        """
        pass

    @abc.abstractmethod
    def calc_throughput(self) -> float:
        pass

    @abc.abstractmethod
    def calc_usage(self) -> float:
        pass

    @abc.abstractmethod
    def calc_num_jobs_over_time(self) -> float:
        pass


class ResourceMetrics(ISimulatedResourceMetrics):
    # processed_jobs_cnt_total: int
    # processing_time_of_jobs_total: float
    # jobs_cnt_during_time: float
    # total time over which metrics were being gathered/added:
    # total_time: float

    def __init__(self) -> None:
        self.processed_jobs_cnt_total = 0
        self.processing_time_of_jobs_total = 0
        self.jobs_cnt_during_time = 0
        self.total_time = 0

    def add_processing_jobs_cnt(self, count: int):
        self.processed_jobs_cnt_total += count

    def add_processing_time_of_jobs(self, time: float):
        self.processing_time_of_jobs_total += time

    def add_jobs_cnt_during_time(self, count_time: float):
        self.jobs_cnt_during_time += count_time

    def set_total_time_passed(self, duration: float):
        self.total_time = duration
        self._validate_total_time()

    def calc_throughput(self) -> float:
        self._validate_total_time()
        return self.processed_jobs_cnt_total / self.total_time

    def calc_usage(self) -> float:
        self._validate_total_time()
        return self.processing_time_of_jobs_total / self.total_time

    def calc_num_jobs_over_time(self) -> float:
        self._validate_total_time()
        return self.jobs_cnt_during_time / self.total_time

    def _validate_total_time(self):
        if self.total_time <= 0:
            raise ValueError(
                f'Total time passed must be set to positive value.'
            )
