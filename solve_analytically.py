import numpy as np
import csv
import matplotlib.pyplot as plt
import config
from typing import Iterable



def main():
    analyse_relative_throughputs(
        config.K_RANGE,
        config.SERVICE_TIMES[ : -1],
        config.SERVICE_TIMES[-1],
        config.PROBS,
        config.RESOURCES_ALIASES[1 : ],
        config.PATH_REL_FLOWS_RESULTS_ANALYTICALLY,
    )

    analyse_alpha_maxs(
        config.K_RANGE,
        config.SERVICE_TIMES[ : -1],
        config.SERVICE_TIMES[-1],
        config.PROBS,
        config.RESOURCES_ALIASES[1 : ],
    )

    analyse_all(
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES[ : -1],
        config.SERVICE_TIMES[-1],
        config.PROBS,
        config.RESOURCES_ALIASES[1 : ],
        config.PATH_ALL_RESULTS_ANALYTICALLY,
    )

    return



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


def extract_generator_from_probs(probs: np.array):
    alphas_relative = probs[0][1 : ]
    probs = probs[1 : , 1 : ]
    return probs, alphas_relative


def anavlyse(
    num_usr_disks: int,
    alpha_scaling_factor: float,
    ser_times: np.array,
    probs: np.array,
    alphas_relative: np.array,
    resources_aliases: np.array,
):
    # lambdas = (-P.T + I) ^ -1 @ alphas
    # alphas = max_alpha * alphas_relative
    # lambdas = (-P.T + I) ^ -1 @ (max_alpha * alphas_relative)
    # lambdas = max_alpha * lambdas_relative
    # lambdas_relative = (-P.T + I) ^ -1 @ alphas_relative
    lambdas_relative = calc_throughputs(probs, alphas_relative)

    max_alpha, max_alpha_indices = calc_alpha_max_from_throughput(
        ser_times,
        lambdas_relative
    )
    alpha = alpha_scaling_factor * max_alpha
    critical_resources = resources_aliases[max_alpha_indices]

    lambdas = alpha * lambdas_relative
    usages = ser_times * lambdas
    # J = U / (1 - U), U = rho = lambda / mu, mu = 1 / S,
    # iff U_i == 1 then J_i = inf
    nums_jobs = np.copy(usages)
    non_inf_usgs = usages < 1.
    nums_jobs[non_inf_usgs] /= 1. - usages[non_inf_usgs]
    nums_jobs[~non_inf_usgs] = float('inf')

    # after any Di leaves sys, each Di is visited just once
    # v0 = 1 / p04, v1 = p1 * v0, v2 = p2 * v0, ..., vd1 = vd2 ... = 1
    usr_disks_start = probs.shape[0] - num_usr_disks
    prob_leave_sys = 1. - np.sum(probs[0][ : usr_disks_start])
    visits = np.array([ 1 / prob_leave_sys ])
    if num_usr_disks > 0:
        visits = np.append(visits, probs[0][1 : usr_disks_start] * visits[0])
        visits = np.append(visits, np.array([ 1 ] * num_usr_disks))
    recalls = nums_jobs / lambdas
    total_recall = np.sum(visits * recalls)

    return (
        usages, lambdas, nums_jobs, total_recall, critical_resources,
        # alpha, max_alpha, max_alpha_idx,
    )


def calc_alpha_max(
    probs: np.array,
    alphas_relative: np.array,
    serv_times: np.array,
):
    rel_throughputs = calc_throughputs(probs, alphas_relative)
    max_alpha, max_alpha_indices = calc_alpha_max_from_throughput(
        serv_times, rel_throughputs
    )
    return max_alpha, max_alpha_indices


def calc_throughputs(probs: np.array, alphas: np.array):
    # lambdas = (-P.T + I) ^ -1 @ alphas
    lambdas = np.linalg.inv(np.identity(probs.shape[0]) - probs.T) \
            @ alphas
    return lambdas


def calc_alpha_max_from_throughput(serv_times: np.array, lambdas_relative: np.arvray):
    # compute the max_alpha so that each lambda_i <= mu_i = 1 / Si
    # lambdas = max_alpha * lambdas_relative <= 1 / S
    max_alphas = 1 / ser_times / lambdas_relative
    max_alpha_idx = np.argmin(max_alphas)
    max_alpha = max_alphas[max_alpha_idx]
    # add indices of all max_alpha occurrances in max_alphas
    max_alpha_indices = [ max_alpha_idx ]
    for i in range(max_alpha_idx + 1, max_alphas.shape[0]):
        if max_alphas[i] == max_alpha:
            max_alpha_indices.append(i)
    return max_alpha, np.array(max_alpha_indices, dtype=np.uint)


def analyse_relative_throughvputs(
    num_usr_disks_range,
    ser_times_no_usr_disks: list[float],
    ser_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    file_path,
):
    with open(file_path, 'w') as file:
        pass  # clear the file

    for num_usr_disks in num_usr_disks_range:
        ser_times, probs, resources_aliases = add_user_disks(
            num_usr_disks,
            ser_times_no_usr_disks,
            ser_time_usr_disk,
            probs_no_usr_disks,
            resources_aliases_no_usr_disks,
        )
        probs, alphas_relative = extract_generator_from_probs(probs)

        headers = [ 'K' ]
        headers += [ f'Rel Throughput {alias}' for alias in resources_aliases ]

        rel_throughputs = calc_throughputs(probs, alphas_relative)

        with open(file_path, 'a') as file:
            csv_writer = csv.writer(file, delimiter='\t')
            csv_writer.writerow([])
            csv_writer.writerow(headers)
            csv_writer.writerow(
                [  num_usr_disks ] + [ f'{el:.3f}' for el in rel_throughputs ]
            )

    print(f'\nStored relative throughputs results to file:\n{file_path}\n')

    return


def analyse_alpha_vmaxs(
    num_usr_disks_range,
    ser_times_no_usr_disks: list[float],
    ser_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    max_alphas = []
    for num_usr_disks in num_usr_disks_range:
        ser_times, probs, resources_aliases = add_user_disks(
            num_usr_disks,
            ser_times_no_usr_disks,
            ser_time_usr_disk,
            probs_no_usr_disks,
            resources_aliases_no_usr_disks,
        )
        probs, alphas_relative = extract_generator_from_probs(probs)

        max_alpha, critical_resources_indices = calc_alpha_max(
            probs,
            alphas_relative,
            ser_times,
        )
        max_alphas.append(max_alpha)

        print(f'\nK: {num_usr_disks}')
        print(f'Critical resources: {resources_aliases[critical_resources_indices]}')

    ax.plot(num_usr_disks_range, max_alphas, '-o')
    ax.set_xlabel('K')
    ax.set_xticks([ val for val in num_usr_disks_range ])
    ax.set_ylabel('alpha_max')
    plt.title(f'alpha_max(K)')
    plt.show()


def analyse_all(
    num_usr_disks_range,
    alpha_scaling_factors_range,
    serv_times_no_usr_disks: list[float],
    serv_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    file_path,
):
    with open(file_path, 'w') as file:
        pass  # clear the file

    for num_usr_disks in num_usr_disks_range:
        for alpha_scaling_factor in alpha_scaling_factors_range:
            serv_times, probs, resources_aliases = add_user_disks(
                num_usr_disks,
                serv_times_no_usr_disks,
                serv_time_usr_disk,
                probs_no_usr_disks,
                resources_aliases_no_usr_disks,
            )
            probs, alphas_relative = extract_generator_from_probs(probs)

            ( usages, throughputs, nums_jobs, total_recall,
              critical_resources,
            ) = analyse(
                num_usr_disks,
                alpha_scaling_factor,
                serv_times,
                probs,
                alphas_relative,
                resources_aliases
            )

            headers = [ 'Metric' ]
            headers += [ f'{alias}' for alias in resources_aliases ]

            with open(file_path, 'a') as file:
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

    print(f'\nStored all results to file:\n{file_path}\n')

    return



if __name__ == '__main__':
    main()
