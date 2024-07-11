import abc
from simulation.network.network import INetwork
from simulation.jobs.job import IJob


class INetworkMetrics(abc.ABC):
    @abc.abstractmethod
    def add_recall_time(
        self,
        total_recall_time: float,
        total_num_jobs: int,
    ):
        pass

    @abc.abstractmethod
    def calc_avg_recall_time(self) -> float:
        pass

    @abc.abstractmethod
    def get_num_jobs_left_sys(self) -> int:
        pass


class NetworkMetrics(INetworkMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.total_recall_time = 0
        self.total_num_jobs = 0

    def add_recall_time(
        self,
        total_recall_time: float,
        total_num_jobs: int,
    ):
        self.total_recall_time += total_recall_time
        self.total_num_jobs += total_num_jobs

    def calc_avg_recall_time(self) -> float:
        return self.total_recall_time / self.total_num_jobs

    def get_num_jobs_left_sys(self) -> int:
        return self.total_num_jobs


class NetworkMetricsFactory:
    # _net: INetwork

    def __init__(self, net: INetwork) -> None:
        self._net = net

    def create_metrics(self) -> NetworkMetrics:
        metrics = NetworkMetrics()

        def add_recall_time(job: IJob, left_net_time: float):
            recall_time = left_net_time - job.created_at()
            metrics.add_recall_time(recall_time, 1)

        self._net.subscribe(
            INetwork.Event.ON_JOB_LEAVE_NETWORK,
            add_recall_time
        )
