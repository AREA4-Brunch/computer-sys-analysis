import os
import numpy as np
import csv
from typing import Iterable


def extract_generator_from_probs(probs: np.array):
    alphas_relative = probs[0][1 : ]
    probs = probs[1 : , 1 : ]
    return probs, alphas_relative


def add_user_disks(
    num_usr_disks: int,
    serv_times_no_usr_disks: Iterable[float],
    serv_time_usr_disk: float,
    probs_no_usr_disks: Iterable[float],
    resources_aliases_no_disks: Iterable[str],
) -> tuple[ np.array, np.array, np.array ]:
    """ Assumes last given serv_time corresponds to first user disk
        and is also same for all other user disks.
    """
    if num_usr_disks <= 0:
        return (
            np.array(serv_times_no_usr_disks),
            np.array(probs_no_usr_disks),
            np.array(resources_aliases_no_disks),
        )

    resources_aliases = [ alias for alias in resources_aliases_no_disks ]
    resources_aliases += [ f'Usr disk {i + 1}' for i in range(num_usr_disks) ]
    resources_aliases = np.array(resources_aliases)

    # add serv_times for each usr disk, 1st disk serv_time is already there
    serv_times = np.array(serv_times_no_usr_disks + [ serv_time_usr_disk ] * num_usr_disks)

    # copy probs_ and add usr disks as destinations probs to
    # existing srcs rows, each usr disk with same resisting prob
    probs = [
          [ el for el in row ]
        + [ (1 - sum(row)) / num_usr_disks ] * num_usr_disks
        for row in probs_no_usr_disks
    ]
    # add usr disks as new srcs probs, jobs exit sys after any usr disk
    probs += num_usr_disks * [
        [ 0 ] * (len(probs_no_usr_disks) + num_usr_disks)
    ]
    probs = np.array(probs)

    return serv_times, probs, resources_aliases


def store_results(
    num_usr_disks: int,
    alpha_scaling_factor: float,
    resources_aliases: Iterable[str],
    critical_resources: Iterable[str],
    recall_time_network: float,
    usages: Iterable[float],
    throughputs: Iterable[float],
    nums_jobs: Iterable[float],
    file_path,
    file_open_mode='a',
):
    headers = [ 'Metric' ]
    headers += [ f'{alias}' for alias in resources_aliases ]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, file_open_mode) as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow([])
        csv_writer.writerow([ 'K:', num_usr_disks ])
        csv_writer.writerow([ 'R:', alpha_scaling_factor ])
        csv_writer.writerow([ 'Network\'s Recall Time [sec]:', f'{(recall_time_network / 1000.):.3f}' ])
        csv_writer.writerow([ 'Critical resources:' ] + [ cr for cr in critical_resources ])
        csv_writer.writerow(headers)

        csv_writer.writerow(
            [ 'Usages:' ] + [ f'{el:.3f}' for el in usages ]
        )
        csv_writer.writerow(
            [ 'Throughputs [jobs/sec]:' ] + [ f'{(1000. * el):.3f}' for el in throughputs ]
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

