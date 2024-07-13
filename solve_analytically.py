import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import config
from common import (
    add_user_disks,
    store_results,
    extract_generator_from_probs,
)


PATH_CWD = os.path.dirname(os.path.realpath(__file__))



def main():
    analyse_relative_throughputs(
        config.K_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES[1 : ],
        os.path.join(PATH_CWD, config.PATH_REL_FLOWS_RESULTS_ANALYTICALLY),
    )

    analyse_alpha_maxs(
        config.K_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES[1 : ],
        os.path.join(PATH_CWD, config.PATH_ALPHA_MAX_PLOT),
    )

    analyse_all(
        config.K_RANGE,
        config.R_RANGE,
        config.SERVICE_TIMES_MS[ : -1],
        config.SERVICE_TIMES_MS[-1],
        config.PROBS,
        config.RESOURCES_ALIASES[1 : ],
        os.path.join(PATH_CWD, config.PATH_ALL_RESULTS_ANALYTICALLY),
    )

    return




def analyse(
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
    recall_network = np.sum(visits * recalls)

    return (
        usages, lambdas, nums_jobs, recall_network, critical_resources,
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


def calc_alpha_max_from_throughput(
    serv_times: np.array,
    lambdas_relative: np.array
):
    # compute the max_alpha so that each lambda_i <= mu_i = 1 / Si
    # lambdas = max_alpha * lambdas_relative <= 1 / S
    max_alphas = 1 / serv_times / lambdas_relative
    max_alpha_idx = np.argmin(max_alphas)
    max_alpha = max_alphas[max_alpha_idx]
    # add indices of all max_alpha occurrances in max_alphas
    max_alpha_indices = [ max_alpha_idx ]
    for i in range(max_alpha_idx + 1, max_alphas.shape[0]):
        if max_alphas[i] == max_alpha:
            max_alpha_indices.append(i)
    return max_alpha, np.array(max_alpha_indices, dtype=np.uint)


def analyse_relative_throughputs(
    num_usr_disks_range,
    ser_times_no_usr_disks: list[float],
    ser_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    file_path,
):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        pass  # clear the file

    # display all at once so all data would share one header with
    # all resource aliases which are constructed at the end
    rels_throughputs_output = []

    for num_usr_disks in num_usr_disks_range:
        serv_times, probs, resources_aliases = add_user_disks(
            num_usr_disks,
            ser_times_no_usr_disks,
            ser_time_usr_disk,
            probs_no_usr_disks,
            resources_aliases_no_usr_disks,
        )
        probs, alphas_relative = extract_generator_from_probs(probs)
        rel_throughputs = calc_throughputs(probs, alphas_relative)

        rels_throughputs_output.append((num_usr_disks, rel_throughputs))

    headers = [ 'K - num of user disks' ]  # use only the last one's aliases
    headers += [ f'{alias}' for alias in resources_aliases ]
    with open(file_path, 'a') as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow(headers)
        for (num_usr_disks, rel_throughputs) in rels_throughputs_output:
            csv_writer.writerow(
                [  num_usr_disks ] + [ f'{el:.3f}' for el in rel_throughputs ]
            )

    print(f'\nStored relative throughputs results to file:\n{file_path}\n')
    return


def analyse_alpha_maxs(
    num_usr_disks_range,
    ser_times_no_usr_disks: list[float],
    ser_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    plt_file_path: str=None,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    max_alphas = []
    for num_usr_disks in num_usr_disks_range:
        serv_times, probs, resources_aliases = add_user_disks(
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
            serv_times,
        )
        max_alphas.append(max_alpha)

        print(f'\nK: {num_usr_disks}')
        critical_resources = resources_aliases[critical_resources_indices]
        print(f'Critical resources: {critical_resources}')

    ax.plot(num_usr_disks_range, max_alphas, '-o')
    ax.set_xlabel('K - num user disks added')
    ax.set_xticks([ val for val in num_usr_disks_range ])
    ax.set_ylabel('alpha_max')
    plt.title(f'alpha_max(K)')
    if plt_file_path is not None:
        plt.savefig(plt_file_path)
        print(f'\nSaved alpha_max figure to file:\n{plt_file_path}\n')
    plt.show()


def analyse_all(
    num_usr_disks_range,
    alpha_scaling_factors_range,
    serv_times_no_usr_disks: list[float],
    serv_time_usr_disk: float,
    probs_no_usr_disks: list[float],
    resources_aliases_no_usr_disks: list[str],
    results_file_path,
):
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    with open(results_file_path, 'w') as file:
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

            ( usages, throughputs, nums_jobs, network_recall,
              critical_resources,
            ) = analyse(
                num_usr_disks,
                alpha_scaling_factor,
                serv_times,
                probs,
                alphas_relative,
                resources_aliases
            )

            store_results(
                num_usr_disks,
                alpha_scaling_factor,
                resources_aliases,
                critical_resources,
                network_recall,
                usages,
                throughputs,
                nums_jobs,
                results_file_path,
            )

    print(f'\nStored all results to file:\n{results_file_path}\n')

    return



if __name__ == '__main__':
    main()
