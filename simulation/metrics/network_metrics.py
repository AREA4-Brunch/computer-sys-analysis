import abc
from ..network.network import INetwork
from ..jobs.job import IJob


class INetworkMetrics(abc.ABC):
    @abc.abstractmethod
    def add_recall_time(
        self,
        total_recall_time: float,
        total_num_jobs: int,
    ):
        pass

    @abc.abstractmethod
    def calc_recall_time(self) -> float:
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
        recall_time: float,
        num_jobs: int,
    ):
        self.total_recall_time += recall_time
        self.total_num_jobs += num_jobs

    def calc_recall_time(self) -> float:
        if self.total_num_jobs == 0: return 0;  # avoid div by 0
        return self.total_recall_time / self.total_num_jobs

    def get_num_jobs_left_sys(self) -> int:
        return self.total_num_jobs

    def register_to_net(self, net: INetwork):
        net.subscribe(
            INetwork.Event.ON_JOB_LEAVE_NETWORK,
            self._on_job_leave_net
        )

    def _on_job_leave_net(self, job: IJob, left_net_time: float):
        recall_time = left_net_time - job.created_at()
        self.add_recall_time(recall_time, 1)
