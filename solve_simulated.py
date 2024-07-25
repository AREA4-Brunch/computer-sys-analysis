import os
import numpy as np
import logging
import time
import multiprocessing
import config
from typing import Iterable, Callable
from simulation.simulation import (
    ISimulation,
    Simulation,
    SequentialSimulations,
    MultiProcessedSimulations,
)
from simulation.core.timer import ITimer
from simulation.core.scheduler import ITasksScheduler
from simulation.resources.resource_interfaces import ISimulatedResource
from simulation.resources.resource_factories import (
    MetricsTrackingResourceFactory,
    MetricsTrackingConfig,
)
from simulation.metrics.resource_metrics import ISimulatedResourceMetrics, ResourceMetrics
from simulation.network.network import SimulatedNetwork
from simulation.metrics.network_metrics import INetworkMetrics, NetworkMetrics
from common import (
    add_user_disks,
    store_results,
    extract_generator_from_probs,
    create_or_clear_file,
)
from solve_analytically import calc_alpha_max



def main():
    path_cwd = os.path.dirname(os.path.realpath(__file__))

    # run once for all (K, R) and store results
    _exec_and_measure_time(
        f'simulate_all (num_reruns=1)',
        simulate_all,
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES,
        os.path.join(path_cwd, config.PATH_SIMULATION_RESULTS),
        num_reruns=1,
        # cause service times are in [ms]
        sim_max_duration=config.SIMULATION_DURATION_MINS * 60 * 1000,
        seed=None,
    )

    # avg out results of multiple parallel reruns for all (K, R) and store results
    _exec_and_measure_time(
        f'simulate_all (num_reruns={config.SIMULATION_AVG_NUM_RERUNS})',
        simulate_all,
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES,
        os.path.join(path_cwd, config.PATH_SIMULATIONS_AVG_RESULTS),
        num_reruns=config.SIMULATION_AVG_NUM_RERUNS,
        # cause service times are in [ms]
        sim_max_duration=config.SIMULATION_DURATION_MINS * 60 * 1000,
        seed=None,
    )

    return



def simulate_all(
    num_usr_disks_range,
    alpha_scaling_factors_range,
    serv_times_no_usr_disks: list[float],
    serv_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    results_file_path,
    num_reruns: int,
    sim_max_duration: float,
    seed: int | None=None,
):
    logger = _create_logger('main', logging.DEBUG)
    if seed is None:  # create seed if not given
        seed = np.random.randint(int(1e5))
        logger.info(
            f'simulate_all chose a random starting seed: {seed}'
          + f' for all K, R combos and NUM_RERUNS: {num_reruns}'
        )

    create_or_clear_file(results_file_path)
    for num_usr_disks in num_usr_disks_range:
        for alpha_scaling_factor in alpha_scaling_factors_range:
            logger.info('\n' + '=' * 50)
            logger.info(f'K: {num_usr_disks}, R: {alpha_scaling_factor}')
            logger.info(f'NUM_RERUNS: {num_reruns}')
            logger.info(f'Seed: {seed}')
            logger.info('=' * 50 + '\n')
            # add user disks to original input data
            serv_times, probs, resources_aliases = add_user_disks(
                num_usr_disks,
                serv_times_no_usr_disks,
                serv_time_usr_disk,
                probs_no_usr_disks,
                resources_aliases_no_usr_disks,
            )
            simulate_one_net_combo(
                alpha_scaling_factor,
                serv_times,
                probs,
                resources_aliases,
                num_reruns,
                num_usr_disks,
                results_file_path,
                sim_max_duration,
                seed,
            )


def simulate_one_net_combo(
    alpha_scaling_factor: float,
    service_times: np.array,
    probs: Iterable[float],
    resources_aliases: Iterable[str],
    num_reruns: int,
    num_usr_disks: int,  # for showing that number in results
    results_file_path,
    max_duration: float,
    seed: int=0,
):
    max_alpha, critical_resources_indices = calc_alpha_max(
        *extract_generator_from_probs(probs),
        service_times,
    )
    alpha = alpha_scaling_factor * max_alpha
    usages, throughputs, nums_jobs, recall_time = simulate(
        alpha,
        service_times,
        probs,
        service_times.shape[0],  # num_resources
        num_reruns,
        max_duration,
        seed,
    )
    # idx+1 cause critical_resources_indices skips generator in aliases
    critical_resources = [
        resources_aliases[idx + 1]
        for idx in critical_resources_indices
    ]
    store_results(
        num_usr_disks,
        alpha_scaling_factor,
        resources_aliases,
        critical_resources,
        recall_time,
        usages,
        throughputs,
        nums_jobs,
        results_file_path,
    )


def simulate(
    alpha: float,
    service_times: Iterable[float],
    probs: Iterable[float],
    num_resources: int,
    num_reruns: int,
    max_duration: float,
    seed: int=0,
    max_workers: int | None=None,  # None for max cores available
) -> tuple[list[float], list[float], list[float], float]:
    simulation = (
        SequentialSimulations() if num_reruns <= 1
        else MultiProcessedSimulations()
    )
    for _ in range(num_reruns):
        simulation.add(Simulation())
    manager = multiprocessing.Manager()
    sim_shared_vars_lock = manager.Lock()
    # wrap each shared int/float as shared list instead of manager.Value so interface
    # would not require `.value` and no refactoring needed in case of no parallelism
    # or multi threaded where they only need to be mutable
    seed            = manager.list([ seed ])
    net_idx         = manager.list([ 0 ])
    avg_recall_time = manager.list([ 0 ])
    avg_usages      = manager.list([ 0 for _ in range(num_resources + 1) ])
    avg_throughputs = manager.list([ 0 for _ in range(num_resources + 1) ])
    avg_nums_jobs   = manager.list([ 0 for _ in range(num_resources + 1) ])
    on_sim_start: Callable[[ISimulation], None] = OnSimulationStart(
        seed=seed,
        net_idx=net_idx,
        num_resources=num_resources,
        alpha=alpha,
        service_times=service_times,
        probs=probs,
        avg_recall_time=avg_recall_time,
        avg_usages=avg_usages,
        avg_throughputs=avg_throughputs,
        avg_nums_jobs=avg_nums_jobs,
        lock=sim_shared_vars_lock,
    )
    simulation.subscribe(ISimulation.Event.ON_START, on_sim_start)
    simulation.simulate(max_duration)
    # avg results out over the number of simulations
    avg_recall_time = avg_recall_time[0] / num_reruns
    avg_usages      = [ usage / num_reruns for usage in avg_usages ]
    avg_throughputs = [ throughput / num_reruns for throughput in avg_throughputs ]
    avg_nums_jobs   = [ num_jobs / num_reruns for num_jobs in avg_nums_jobs ]
    return avg_usages, avg_throughputs, avg_nums_jobs, avg_recall_time


def _create_logger(name, lvl) -> logging.Logger:
    logger = logging.Logger(name, lvl)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(lvl)
    logger.addHandler(console_handler)
    return logger


def _exec_and_measure_time(msg: str, func: callable, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    print(f'Total execution time of {msg}: {duration}')


class OnSimulationStart:
    """ Workaround since `concurrent` uses pickle which cannot serialize
        local functions, so moving in all local params of what could be
        local func into this object.
    """

    def __init__(
        self,
        seed: list[int],  # single int wrapped in list to be mutable
        net_idx: list[int],  # single int wrapped in list to be mutable
        num_resources: int,
        alpha: float,
        service_times: Iterable[float],
        probs: Iterable[float],
        avg_recall_time: list[float],  # single float wrapped in list to be mutable
        avg_usages: list[float],
        avg_throughputs: list[float],
        avg_nums_jobs: list[float],
        lock: multiprocessing.Lock,
    ):
        self._seed = seed
        self._net_idx = net_idx
        self._num_resources = num_resources
        self._alpha = alpha
        self._service_times = service_times
        self._probs = probs
        self._avg_recall_time = avg_recall_time
        self._avg_usages = avg_usages
        self._avg_throughputs = avg_throughputs
        self._avg_nums_jobs = avg_nums_jobs
        self._lock = lock

    def __call__(self, sim: ISimulation):
        self.on_simulation_start(sim)

    def on_simulation_start(self, sim: ISimulation):
        # pull all data needed and unsubscribe since
        # each simulation can be started only once
        timer: ITimer = sim.timer()
        scheduler: ITasksScheduler = sim.scheduler()
        sim.unsubscribe(ISimulation.Event.ON_START, self)
        with self._lock:
            self._seed[0] += 1; self._net_idx[0] += 1;
            psrng = np.random.RandomState(seed=self._seed[0])
            resources, resource_metrics = self._create_resources(timer, psrng)
            net, net_metrics, logger = self._create_net(resources, psrng, timer)
        # no more modifications on self, attrs used below are const ptrs
        real_time_sim_start = time.time()
        on_sim_end: Callable[[ISimulation], None] = OnSimulationEnd(
            avg_recall_time=self._avg_recall_time,
            avg_usages=self._avg_usages,
            avg_throughputs=self._avg_throughputs,
            avg_nums_jobs=self._avg_nums_jobs,
            net_metrics=net_metrics,
            resource_metrics=resource_metrics,
            real_time_sim_start=real_time_sim_start,
            logger=logger,
            sim_idx=self._net_idx[0],
            lock=self._lock,
        )
        sim.subscribe(ISimulation.Event.ON_END, on_sim_end)
        net.start(scheduler)

    def _create_resources(self, timer: ITimer, psrng: np.random.RandomState):
        resource_factory = MetricsTrackingResourceFactory()
        resource_metrics = [ ResourceMetrics() for _ in range(self._num_resources + 1) ]
        resources = [  # add job generator
            resource_factory.create_job_generator(
                MetricsTrackingConfig(
                    to_track={ 'usage', 'throughput', 'num_jobs' },
                    timer=timer,
                    metrics_registry=resource_metrics[0],
                ),
                alpha=self._alpha,
                timer=timer,
                psrng=psrng,
            )
        ]
        resources += [  # add resources (CPUs and disks)
            resource_factory.create_std_resource(
                MetricsTrackingConfig(
                    to_track={ 'usage', 'throughput', 'num_jobs' },
                    timer=timer,
                    metrics_registry=resource_metrics[idx + 1],
                ),
                serv_time=serv_time,
            )
            for idx, serv_time in enumerate(self._service_times)
        ]

        return resources, resource_metrics

    def _create_net(
        self,
        resources: list[ISimulatedResource],
        psrng: np.random.RandomState,
        timer: ITimer,
    ):
        net_name = f'NETWORK-{self._net_idx[0]}'
        logger = _create_logger(f'LOGGER-{net_name}', logging.DEBUG)
        net = SimulatedNetwork(
            resources,
            self._probs,
            psrng,
            timer,
            logger,
            net_name,
        )
        net_metrics = NetworkMetrics()
        net_metrics.register_to_net(net)
        return net, net_metrics, logger


class OnSimulationEnd:
    """ Workaround since `concurrent` uses pickle which cannot serialize
        local functions, so moving in all local params of what could be
        local func into this object.
    """

    def __init__(
        self,
        avg_recall_time: list[float],  # single float wrapped in list to be mutable
        avg_usages: list[float],
        avg_throughputs: list[float],
        avg_nums_jobs: list[float],
        net_metrics: INetworkMetrics,
        resource_metrics: list[ISimulatedResourceMetrics],
        real_time_sim_start: float,
        logger: logging.Logger,
        sim_idx: int,
        lock: multiprocessing.Lock,
    ):
        self._avg_recall_time = avg_recall_time
        self._avg_usages = avg_usages
        self._avg_throughputs = avg_throughputs
        self._avg_nums_jobs = avg_nums_jobs
        self._net_metrics = net_metrics
        self._resource_metrics = resource_metrics
        self._real_time_sim_start = real_time_sim_start
        self._logger = logger
        self._sim_idx = sim_idx
        self._lock = lock

    def __call__(self, sim: ISimulation):
        self.on_simulation_end(sim)

    def on_simulation_end(self, sim: ISimulation):
        duration = sim.duration()
        sim.unsubscribe(ISimulation.Event.ON_END, self)
        # calc results of the simulation
        with self._lock:
            self._avg_recall_time[0] += self._net_metrics.calc_recall_time()
            for idx, metric in enumerate(self._resource_metrics):
                metric.add_total_time_passed(duration)
                self._avg_usages[idx] += metric.calc_usage()
                self._avg_throughputs[idx] += metric.calc_throughput()
                self._avg_nums_jobs[idx] += metric.calc_num_jobs_over_time()
            # log real time and sim time duration of the simulation:
            real_time_sim_end = time.time()
            real_time_duration = real_time_sim_end - self._real_time_sim_start
            self._logger.info(f'SIMULATION #{self._sim_idx} IS OVER')
            self._logger.info(f'NUM JOBS LEFT SYS: {self._net_metrics.get_num_jobs_left_sys()}')
            self._logger.info(f'REAL TIME DURATION [sec]: {real_time_duration}')
            self._logger.info(f'SIMULATED TIME DURATION [sec]: {duration / 1000}')



if __name__ == '__main__':
    main()
