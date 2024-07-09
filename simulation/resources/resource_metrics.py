from simulation.jobs.job import IJob
from simulation.resources.resource_interfaces import (
    IResource,
)


class ProcessingTime(IResource):
    # _resource: IResource
    # processed_jobs_cnt_total: int
    # processing_time_of_jobs_total: float

    def __init__(self, subject_resource: IResource) -> None:
        super().__init__()
        self._resource = subject_resource
        self.processed_jobs_cnt_total = 0
        self.processing_time_of_jobs_total = 0

    def insert_job(self, job: IJob):
        self._resource.insert_job(job)

    def get_cur_job_eta(self) -> float:
        return self._resource.get_cur_job_eta()

    def process_cur_job(self) -> IJob:
        self.processed_jobs_cnt_total += 1
        job, arrival_time = self._resource.process_cur_job()
        processing_time = self._get_cur_time() - arrival_time
        self.processing_time_of_jobs_total += processing_time
        return job

    def _get_cur_time(self) -> float:
        return self._resource._get_cur_time()
