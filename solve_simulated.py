import os
import numpy as np
import logging
import time  # for debugging
import config
from typing import Iterable
from simulation.simulation.simulation import (
    ISimulation,
    Simulation,
    ConsecutiveSimulations,
)
from simulation.utils.timer import ITimer
from simulation.simulation.scheduler import ITasksScheduler
from simulation.resources.resource_factories import (
    MetricsTrackingResourceFactory,
    MetricsTrackingFactoryMethodConfig,
)
from simulation.metrics.resource_metrics import ResourceMetrics
from simulation.network.network import SimulatedNetwork
from simulation.metrics.network_metrics import NetworkMetrics
from common import (
    add_user_disks,
    store_results,
    extract_generator_from_probs,
)
from solve_analytically import calc_alpha_max


PATH_CWD = os.path.dirname(os.path.realpath(__file__))



def main():
    _exec_and_measure_time(
        f'sim_all (num_reruns=1)',
        simulate_all,
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES,
        os.path.join(PATH_CWD, config.PATH_SIMULATION_RESULTS),
        num_reruns=1,
        # cause service times are in [ms]
        sim_max_duration=config.SIMULATION_DURATION_MINS * 60 * 1000,
        seed=None,
    )

    _exec_and_measure_time(
        f'sim_all (num_reruns={config.SIMULATION_AVG_NUM_RERUNS})',
        simulate_all,
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES,
        os.path.join(PATH_CWD, config.PATH_SIMULATIONS_AVG_RESULTS),
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
    if seed is None:
        seed = np.random.randint(int(1e5))
        logger.info(
            f'simulate_all chose a random starting seed: {seed}'
          + f' for all K, R combos and NUM_RERUNS: {num_reruns}'
        )

    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with open(results_file_path, 'w') as file:
        pass  # clear the file

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

    # critical_resources_indices skips the generator in aliases
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
) -> tuple[list[float], list[float], list[float], float]:
    simulation = ConsecutiveSimulations()
    for _ in range(num_reruns):
        simulation.add(Simulation())

    resource_factory = MetricsTrackingResourceFactory()
    agg_metrics = [ ResourceMetrics() for _ in range(num_resources + 1) ]
    agg_net_metrics = NetworkMetrics()

    net_idx = 0  # for naming the networks
    def on_simulation_start(sim: ISimulation):
        nonlocal seed, net_idx
        seed += 1; net_idx += 1;

        # pull all data needed and unsubscribe since
        # each simulation can be started only once
        timer: ITimer = sim.timer()
        scheduler: ITasksScheduler = sim.scheduler()
        sim.unsubscribe(ISimulation.Event.ON_START, on_simulation_start)

        psrng = np.random.RandomState(seed=seed)
        resources = [  # add job generator
            resource_factory.create_job_generator(
                MetricsTrackingFactoryMethodConfig(
                    to_track={ 'usage', 'throughput', 'num_jobs' },
                    timer=timer,
                    metrics_registry=agg_metrics[0],
                ),
                alpha=alpha,
                timer=timer,
                psrng=psrng,
            )
        ]
        resources += [  # add resources (CPUs and disks)
            resource_factory.create_std_resource(
                MetricsTrackingFactoryMethodConfig(
                    to_track={ 'usage', 'throughput', 'num_jobs' },
                    timer=timer,
                    metrics_registry=agg_metrics[idx + 1],
                ),
                serv_time=serv_time,
            )
            for idx, serv_time in enumerate(service_times)
        ]

        net_name = f'NETWORK-{net_idx}'
        logger = _create_logger(f'LOGGER-{net_name}', logging.DEBUG)
        net = SimulatedNetwork(
            resources,
            probs,
            psrng,
            timer,
            logger,
            net_name,
        )
        agg_net_metrics.register_to_net(net)

        real_time_sim_start = time.time()
        def on_simulation_end(sim: ISimulation):
            duration = sim.duration()
            simulation.unsubscribe(ISimulation.Event.ON_END, on_simulation_end)
            for metric in agg_metrics:
                metric.add_total_time_passed(duration)

            # log real time and sim time duration of the simulation:
            real_time_sim_end = time.time()
            real_time_duration = real_time_sim_end - real_time_sim_start
            logger.info(f'REAL TIME DURATION [sec]: {real_time_duration}')
            logger.info(f'SIMULATED TIME DURATION [sec]: {duration / 1000}')

        simulation.subscribe(ISimulation.Event.ON_END, on_simulation_end)
        net.start(scheduler)

    simulation.subscribe(ISimulation.Event.ON_START, on_simulation_start)
    simulation.simulate(max_duration)

    # calc results of the simulation
    recall_time = agg_net_metrics.calc_recall_time()
    usages, throughputs, nums_jobs = [], [], []
    for metric in agg_metrics:
        usages.append(metric.calc_usage())
        throughputs.append(metric.calc_throughput())
        nums_jobs.append(metric.calc_num_jobs_over_time())

    return usages, throughputs, nums_jobs, recall_time


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



if __name__ == '__main__':
    main()
