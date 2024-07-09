import numpy as np
import bisect
import heapq
import abc
import time
import logging
import csv
import config
import multiprocessing
import concurrent.futures
import gc
import traceback
from collections import deque, defaultdict
from typing import Iterable, Callable
from solve_analytically import (
    add_user_disks,
    extract_generator_from_probs,
    calc_alpha_max,
)
from simulation.utils.timer import ITimer
from simulation.jobs.job import IJob



class ResourcesMetricsAggregator(ResourceMetricsCalculator):
    """ Thread safe aggregator of results gathered by other
        metric calculators.
    """
    def __init__(self) -> None:
        super().__init__()
        self._lock = multiprocessing.Lock()

    def on_calc_recording_end(self, calc: ResourceMetricsCalculator):
        with self._lock:
            self.processed_jobs_cnt_total += calc.processed_jobs_cnt_total
            self.processing_time_of_jobs_total += calc.processing_time_of_jobs_total
            self.num_of_jobs_over_time_total += calc.num_of_jobs_over_time_total
            self.duration += calc.duration

    def calc_metrics(self):
        with self._lock:
            return super().calc_metrics()


class SimulationMetricsCalculator(abc.ABC):
    def __init__(self):
        self.num_reruns = 0
        self.total_duration = 0
        self.total_recall = 0
        self.total_num_jobs_left_sys = 0

    def calc_metrics(self):
        if self.num_reruns == 0: return 0, 0, 0;
        avg_duration = self.total_duration / self.num_reruns
        avg_total_recall = self.total_recall / self.num_reruns
        avg_num_jobs = self.total_num_jobs_left_sys / self.num_reruns
        return avg_duration, avg_total_recall, avg_num_jobs


class SimulationsMetricsAggregator(SimulationMetricsCalculator):
    """ Thread safe aggregator of results gathered by simulations.
    """
    def __init__(self) -> None:
        super().__init__()
        self._lock = multiprocessing.Lock()

    def on_simulation_end(
        self,
        simulation_duration: float,
        avg_recall_time: float,
        num_jobs_left_sys: int,
        *args,
        **kwargs,
    ):
        with self._lock:
            self.num_reruns += 1
            self.total_duration += simulation_duration
            self.total_recall += avg_recall_time
            self.total_num_jobs_left_sys += num_jobs_left_sys

    def calc_metrics(self):
        with self._lock:
            return super().calc_metrics()


class ObservableResource(Observable, IResource):
    class Events:
        ON_PROCESSED_JOB = 'on_processed_job'

    def __init__(self):
        super().__init__()


class JobGenerator(ObservableResource):
    def __init__(
        self,
        alpha: float,
        timer: ITimer,
        psrng: np.random.RandomState,
    ):
        super().__init__()
        self._alpha = alpha
        self._psrng = psrng
        self._timer = timer
        self.insert_job(self._gen_new_job())

    def insert_job(self, job: IJob):
        self._cur_job_arrival_time = self._timer._now()
        self._cur_job = job
        self._cur_eta = self._gen_eta()

    def get_cur_job_eta(self) -> float:
        return self._cur_eta

    def process_cur_job(self) -> IJob:
        cur_job = self._cur_job
        self._on_processed_job(self._cur_job_arrival_time)
        # move on to the next job
        self.insert_job(self._gen_new_job())
        return cur_job

    def _gen_eta(self):
        generation_duration = self._psrng.exponential(1 / self._alpha)
        eta = self._timer._now() + generation_duration
        return eta

    def _gen_new_job(self) -> IJob:
        return Job(self._timer._now())

    def _on_processed_job(self, procesed_job_arrival_time: float):
        # notify metrics
        metrics_event_name = ObservableResource.Events.ON_PROCESSED_JOB
        if self._num_subscribers(metrics_event_name) > 0:
            job_processing_time = self._timer._now() - procesed_job_arrival_time
            metrics = {
                'processing_time_of_jobs_incr': job_processing_time,
                # only 1 job is at the generator at a time
                'num_of_jobs_over_time_incr': job_processing_time * 1,
            }
            self._notify(metrics_event_name, **metrics)


class Network:
    # _resources: list[IResource]
    # _probs: list[tuple[float, int]] - probs[src] = [ (prob, dst)... ]
    # _resources_etas: heapq[tuple[int, int]]
    #                - top el = (min_eta, min_eta_resource_idx)

    def __init__(
        self,
        resources: list[IResource],
        probs: Iterable,
        timer: ITimer,
        psrng: np.random.RandomState,
        logger: logging.Logger=None,
        name: str='',
    ) -> None:
        self._logger = logger
        self._name = name
        self._log_prefix = f'{self._name}'
        self._resources = resources
        self._probs = self._construct_transition_probs(probs)
        self._timer = timer
        self._psrng = psrng
        self._init_resources_etas()

    def _construct_transition_probs(self, probs: Iterable):
        has_not_warned_about_sys_leave = True
        probs_offseted = []
        for src, dst_probs in enumerate(probs):
            cur_probs = []
            for dst, prob in enumerate(dst_probs):
                if prob <= 0: continue;
                if len(cur_probs):
                    cur_probs.append((prob + cur_probs[-1][0], dst))
                else:
                    cur_probs = [ (prob, dst) ]

            probs_offseted.append(cur_probs)

            # check validity of given probabilities
            has_not_warned_about_sys_leave = self._validate_cur_probs(
                cur_probs,
                src,
                has_not_warned_about_sys_leave,
            )

        return probs_offseted

    def _validate_cur_probs(
        self,
        cur_probs: list[float],
        src: int,
        do_log_independent_warning: bool,
    ) -> bool:
        """ Returns if given probs row is valid and expected.
            If invalid raises an exception, if unexptected logs warnings.
        """
        prob_leaving_sys = 1. - cur_probs[-1][0] if len(cur_probs) else 1.

        if prob_leaving_sys < 0:
            raise Exception(f'Given probabilities in each row must be <= 1!')

        if prob_leaving_sys < 1e-8:  # tolerate rounding error
            # provided probabilities sum up to 1
            return True

        if self._logger is None:  # cannot log warnings
            return False

        # provided probabilities sum up to less than 1 and leftover prob
        # will be treated as prob of leaving the system from `src` resource
        if do_log_independent_warning:
            self._logger.warning(
                f'{self._log_prefix} Warning: All leftover probabilities'
              + f'(1 - sum(row_probs)) of each prob matrix row will represent'
              + f' probability of leaving the system.'
            )

        self._logger.warning(
            f'{self._log_prefix} Warning: prob of leaving system from resource #{src}'
            + f' is {prob_leaving_sys:.5f}'
        )

        return False

    def _init_resources_etas(self):
        self._resources_etas = []
        for idx, resource in enumerate(self._resources):
            eta = resource.get_cur_job_eta()
            if eta != float('inf'):
                self._resources_etas.append((eta, idx))
        heapq.heapify(self._resources_etas)

    @property
    def resources(self) -> list[IResource]:
        return self._resources

    def resource(self, resource_idx: int) -> IResource:
        return self._resources[resource_idx]

    def next_resource_idx(self, src: int) -> int:
        """ Returns idx of resource in self._resources for a job
            to go to next.
            If job after src leaves the net returns -1.
        """
        prob = self._psrng.uniform(0., 1.)
        idx = bisect.bisect_right(self._probs[src], (prob, 0))
        if idx >= len(self._probs[src]):
            return -1  # leaves the system after src
        dst = self._probs[src][idx][1]
        return dst

    def pop_min_eta_resource(self) -> tuple[float, int]:
        min_eta, resource_idx = heapq.heappop(self._resources_etas)
        return min_eta, resource_idx

    def update_eta_resource(self, resource_idx, old_eta):
        new_eta = self._resources[resource_idx].get_cur_job_eta()
        if new_eta != old_eta and new_eta != float('inf'):
            heapq.heappush(self._resources_etas, (new_eta, resource_idx))


class Simulation(Observable):
    class Events:
        ON_SIMULATION_END = 'on_simulation_end'


    def __init__(
        self,
        net: Network,
        timer: ITimer,
        logger: logging.Logger,
        name: str,
    ):
        super().__init__()
        self._net = net
        self._timer = timer
        self._logger = logger
        self._name = name
        self._log_prefix = f'{self._name}'

    @property
    def net(self) -> Network:
        return self._net

    def simulate(self, end_time: float):
        simulation_start_time = time.time()
        avg_recall_time = 0
        num_jobs_left_sys = 0
        while self._timer._now() < end_time:
            eta, resource_idx = self._net.pop_min_eta_resource()
            # !important set cur time before process_cur_job() since
            # it will use the self._timer
            self._timer._cur_time = eta  # !important before process_cur_job()
            job: IJob = self._net.resource(resource_idx).process_cur_job()
            # !important update after new eta has been calculated,
            # which happens in process_cur_job()
            self._net.update_eta_resource(resource_idx, old_eta=eta)
            dst_resource_idx = self._net.next_resource_idx(resource_idx)

            if dst_resource_idx >= 0:
                dst_resource = self._net.resource(dst_resource_idx)
                old_dst_eta = dst_resource.get_cur_job_eta()
                dst_resource.insert_job(job)
                self._net.update_eta_resource(dst_resource_idx, old_dst_eta)

            else:  # job leaves the system
                num_jobs_left_sys += 1
                avg_recall_time += eta - job.created_at

        if num_jobs_left_sys != 0:  # avoid division by 0
            avg_recall_time /= num_jobs_left_sys

        duration_sim = self._timer._now()
        self._on_simulation_end(
            duration_sim, avg_recall_time, num_jobs_left_sys
        )
        # real time durations
        simulation_end_time = time.time()
        duration = simulation_end_time - simulation_start_time
        self._logger.info(f'{self._log_prefix} DURATION: {(duration_sim / 1000):.3f} secs')
        self._logger.info(f'{self._log_prefix} REAL TIME DURATION: {duration:.3f} secs')
        self._logger.info(f'{self._log_prefix} Num jobs left the system: {num_jobs_left_sys}')
        self._logger.info(f'{self._log_prefix} Avg cycle time: {(avg_recall_time / 1000):.3f} secs')

        gc.collect()
        return duration_sim, avg_recall_time, num_jobs_left_sys

    def _on_simulation_end(
        self,
        simulation_duration: float,
        avg_recall_time: float,
        num_jobs_left_sys: int,
    ):
        self._notify(
            Simulation.Events.ON_SIMULATION_END,
            simulation_duration,
            avg_recall_time,
            num_jobs_left_sys,
        )


class SimulationsManager:
    # sims_provider: Callable[[], Simulation]
    # resources_metrics_providers[0] = list of provider funcs, each providing a metric calc
    # resources_metrics_providers: list[Callable[[], list[ObservableResourceMetricsCalc]]]
    # sim_metrics_providers: list[Callable[[], SimulationsMetricsCalculator]]

    def __init__(
        self,
        sim_provider: Callable[[], Simulation],
        resources_metrics_providers: list[list[Callable[[], ObservableResourceMetricsCalc]]],
        sim_metrics_providers: list[Callable[[], SimulationsMetricsCalculator]],
        logger: logging.Logger,
    ) -> None:
        self._sims_provider = sim_provider
        self._resource_metrics_providers = resources_metrics_providers
        self._sim_metrics_providers = sim_metrics_providers
        self._logger = logger

    def simulate(
        self,
        num_reruns: int,
        sim_end_time: float
    ) -> tuple[list[SimulationsMetricsCalculator], list[ResourceMetricsCalculator]]:
        if num_reruns < 0:
            raise Exception(f'`num_reruns` must not be negative.')

        sim_metrics = self._fetch_sim_metrics()
        
        ( simulations, sim_metrics, agg_metrics
        ) = self._register_simulation_metrics(num_reruns)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for simulation in simulations:
                futures.append(executor.submit(
                    simulation.simulate, sim_end_time
                ))
            # wait for all to finish
            for future in concurrent.futures.as_completed(futures):
                try:
                    # all return results are supposed to be picked
                    # up by the sim_metrics
                    future.result()
                except Exception as e:
                    traceback.print_exception(e)
                    self._logger.error(f'An error ocurred during simulation: {e}')

        return sim_metrics, agg_metrics

    def _fetch_resource_metrics(self) -> list[list[ObservableResourceMetricsCalc]]:
        metrics = [
            [ provider() for provider in resource_providers ]
            for resource_providers in self._resource_metrics_providers
        ]
        return metrics

    def _fetch_sim_metrics(self) -> list[SimulationsMetricsCalculator]:
        metrics = [ provider() for provider in self._sim_metrics_providers ]
        return metrics

    def _register_all_metrics(self, simulation: Simulation):
        self._register_resource_metrics(simulation.net.resources)
        self._register_simulation_metrics(simulation)

    def _register_resource_metrics(self, resources: list[IResource]):
        for resource in resources:
            for metric_provider in self._resource_metrics_providers:
                metric = metric_provider()
                # unsubscribe unneccessary, waits for resource to be destroyed
                resource.subscribe(
                    ObservableResource.Events.ON_PROCESSED_JOB,
                    metric.on_processed_job,
                )

    def _register_simulation_metrics(self, simulation: Simulation):
        return

    def _link_resources_and_simulation_metrics(
        self,
        sim_metrics: list[SimulationsMetricsCalculator],
        resources_metrics: list[list[ObservableResourceMetricsCalc]],
    ):
        for idx, metric in enumerate(metrics):
            # register each metric to end of simulation,
            # once it executes unsubscribe
            simulation.subscribe(
                Simulation.Events.ON_SIMULATION_END,
                SubscribeExecUnsubscribe(
                    simulation,
                    Simulation.Events.ON_SIMULATION_END,
                    metric.on_recording_end,
                    notifies_until_unsubscribe=1,
                ),
            )
            # register each agg calc to corresponding metric
            metric.subscribe(
                ObservableResourceMetricsCalc.Events.ON_RECORDING_END,
                SubscribeExecUnsubscribe(
                    metric,
                    ObservableResourceMetricsCalc.Events.ON_RECORDING_END,
                    agg_metrics[idx].on_calc_recording_end,
                    notifies_until_unsubscribe=1,
                ),
            )

        return

    def _register_simulation_metrics_(
        self,
        num_reruns
    ) -> tuple[list[Simulation], list[ResourceMetricsCalculator]]:
        simulations = []
        agg_metrics = []
        sim_metrics = [ SimulationsMetricsCalculator() ]
        for rerun_idx in range(num_reruns):
            simulation, metrics = self._sims_provider()
            simulations.append(simulation)

            # first time create aggregators for all metrics
            if len(agg_metrics) == 0:
                agg_metrics = [
                    ResourcesMetricsAggregator() for _ in range(len(metrics))
                ]

            # register simulation metrics to end of each simulation
            for simulation in simulations:
                for sim_metric in sim_metrics:
                    simulation.subscribe(
                        Simulation.Events.ON_SIMULATION_END,
                        SubscribeExecUnsubscribe(
                            simulation,
                            Simulation.Events.ON_SIMULATION_END,
                            sim_metric.on_simulation_end,
                            notifies_until_unsubscribe=1,
                        ),
                    )

            for idx, metric in enumerate(metrics):
                # register each metric to end of simulation,
                # once it executes unsubscribe
                simulation.subscribe(
                    Simulation.Events.ON_SIMULATION_END,
                    SubscribeExecUnsubscribe(
                        simulation,
                        Simulation.Events.ON_SIMULATION_END,
                        metric.on_recording_end,
                        notifies_until_unsubscribe=1,
                    ),
                )
                # register each agg calc to corresponding metric
                metric.subscribe(
                    ObservableResourceMetricsCalc.Events.ON_RECORDING_END,
                    SubscribeExecUnsubscribe(
                        metric,
                        ObservableResourceMetricsCalc.Events.ON_RECORDING_END,
                        agg_metrics[idx].on_calc_recording_end,
                        notifies_until_unsubscribe=1,
                    ),
                )

        return simulations, sim_metrics, agg_metrics


class ObservableSimulationFactory:
    def __init__(
        self,
        alpha: float,
        serv_times: np.array,
        probs: np.array,
        logging_lvl=logging.DEBUG,
    ):
        self._sim_idx = 1
        self._alpha = alpha
        self._serv_times = serv_times
        self._probs = probs
        self._logging_lvl = logging_lvl

    def create_simulation(self, seed=None) -> Simulation:
        sim_name = f'Simulation {self._sim_idx}'
        logger = logging.Logger(f'Logger {sim_name}', self._logging_lvl)
        timer = Timer()
        psrng = self._gen_psrng(seed, sim_name)
        resources = self.build_resources(timer, psrng)
        net = self.build_network(resources, timer, logger, psrng)
        simulation = Simulation(net, timer, logger, sim_name)
        self._sim_idx += 1
        return simulation

    def build_resources(
        self,
        timer: ITimer,
        psrng: np.random.RandomState
    ) -> list[ObservableResource]:
        resources: list[ObservableResource] = (
            [ JobGenerator(self._alpha, timer, psrng) ]
          + [ StandardResource(serv_time, timer) for serv_time in self._serv_times ]
        )
        return resources

    def build_network(
        self,
        resources: list[ObservableResource],
        timer: ITimer,
        logger: logging.Logger,
        psrng: np.random.RandomState,
    ) -> Network:
        net_name = f'NETWORK {self._sim_idx}'
        net = Network(resources, self._probs, timer, psrng, logger, net_name)
        return net

    def _gen_psrng(self, seed=None, sim_name='') -> np.random.RandomState:
        if seed is None:
            seed = np.random.randint(int(1e5))
            self._logger.info(
                f'SimulationFactory chose a random starting seed: {seed}'
              + f' for `{sim_name}`'
            )
        return np.random.RandomState(seed)


def main():
    simulate_all(
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES[ : -1],
        config.SERVICE_TIMES[-1],
        config.PROBS,
        config.RESOURCES_ALIASES,
        config.PATH_SIMULATION_RESULTS,
        num_reruns=1,
        sim_max_duration=config.SIMULATION_DURATION,
        seed=None,
    )

    simulate_all(
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES[ : -1],
        config.SERVICE_TIMES[-1],
        config.PROBS,
        config.RESOURCES_ALIASES,
        config.PATH_SIMULATIONS_AVG_RESULTS,
        num_reruns=config.SIMULATION_AVG_NUM_RERUNS,
        sim_max_duration=config.SIMULATION_DURATION,
        seed=None,
    )



def simulate_all(
    num_usr_disks_range,
    alpha_scaling_factors_range,
    serv_times_no_usr_disks: list[float],
    serv_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    file_path,
    num_reruns: int,
    sim_max_duration: float,
    seed: int=None,
):
    logger = logging.Logger('Common', level=logging.DEBUG)

    if seed is None:
        seed = np.random.randint(int(1e5))
        logger.info(f'simulate_all chose a random starting seed: {seed}')

    with open(file_path, 'w') as file:
        pass  # clear the file

    for num_usr_disks in num_usr_disks_range:
        for alpha_scaling_factor in alpha_scaling_factors_range:
            logger.info(f'K: {num_usr_disks}, R: {alpha_scaling_factor}')
            logger.info(f'Seed: {seed}')
            psrng = np.random.RandomState(seed)
            seed += 1

            ( sim_metrics, agg_metrics, resources_aliases,
              critical_resources_indices,
            ) = simulate_single_combo(
                num_usr_disks,
                alpha_scaling_factor,
                serv_times_no_usr_disks,
                serv_time_usr_disk,
                probs_no_usr_disks,
                resources_aliases_no_usr_disks,
                num_reruns,
                sim_max_duration,
                psrng,
                logger,
            )

            # calc and unpack aggregated avg metrics results
            usages, throughputs, nums_jobs = [], [], []
            for metric_calc in agg_metrics:
                usage, throughput, num_jobs = metric_calc.calc_metrics()
                usages.append(usage)
                throughputs.append(throughput)
                nums_jobs.append(num_jobs)

            ( avg_duration, avg_total_recall, avg_num_jobs
            ) = sim_metrics[0].calc_metrics()

            store_results(
                num_usr_disks,
                alpha_scaling_factor,
                resources_aliases,
                resources_aliases[critical_resources_indices],
                avg_total_recall,
                usages,
                throughputs,
                nums_jobs,
                file_path,
                file_open_mode='a',
            )

    print(f'\nStored all results to file:\n{file_path}\n')

    return


def simulate_single_combo(
    num_usr_disks,
    alpha_scaling_factor,
    serv_times_no_usr_disks: list[float],
    serv_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    num_reruns: int,
    sim_max_duration: float,
    psrng: np.random.RandomState,
    sim_mngr_logger: logging.Logger,
):
    serv_times, probs, resources_aliases = add_user_disks(
        num_usr_disks,
        serv_times_no_usr_disks,
        serv_time_usr_disk,
        probs_no_usr_disks,
        resources_aliases_no_usr_disks,
    )
    max_alpha, critical_resources_indices = calc_alpha_max(
        *extract_generator_from_probs(probs),
        serv_times,
    )
    alpha = alpha_scaling_factor * max_alpha



    sim_provider = ObservableSimulationFactory().create_simulation
    # resources_metrics_provider[resource] = list of metrics
    resources_metrics_provider = [
        [ ResourceMetricsCalculator, ]
        for _ in range(probs.shape[0])
    ]
    # agg_metrics_provider[resource] = list of metrics
    agg_metrics_provider = [
        [ ResourcesMetricsAggregator, ]
        for _ in range(probs.shape[0])
    ]
    sim_metrics_provider = [
        
    ]


    simulation_mngr = SimulationsManager(
        create_simulation, sim_mngr_logger
    )
    sim_metrics, agg_metrics = simulation_mngr.simulate(
        num_reruns, sim_max_duration
    )
    gc.collect()
    return (
        sim_metrics, agg_metrics, resources_aliases,
        critical_resources_indices,
    )



def store_results(
    num_usr_disks: int,
    alpha_scaling_factor: float,
    resources_aliases: list[str],
    critical_resources: list[str],
    total_recall: float,
    usages: Iterable[float],
    throughputs: Iterable[float],
    nums_jobs: Iterable[float],
    file_path,
    file_open_mode='a',
):
    headers = [ 'Metric' ]
    headers += [ f'{alias}' for alias in resources_aliases ]

    with open(file_path, file_open_mode) as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow([])
        csv_writer.writerow([ 'K:', num_usr_disks ])
        csv_writer.writerow([ 'R:', alpha_scaling_factor ])
        csv_writer.writerow([ 'Total Recall Time [s]:', f'{(total_recall / 1000.):.3f}' ])
        csv_writer.writerow([ 'Critical resources:' ] + [ cr for cr in critical_resources ])
        csv_writer.writerow(headers)

        csv_writer.writerow(
            [ 'Usages:' ] + [ f'{el:.3f}' for el in usages ]
        )
        csv_writer.writerow(
            [ 'Throughputs [jobs/s]:' ] + [ f'{(1000. * el):.3f}' for el in throughputs ]
        )
        csv_writer.writerow(
            [ 'Avg num of jobs:' ] + [ f'{el:.3f}' for el in nums_jobs ]
        )

        csv_writer.writerow(
            [ 'Is critical resource:' ]
            + [
                'YES' if (el in critical_resources) else 'NO'
                for el in resources_aliases
            ]
        )



if __name__ == '__main__':
    main()
