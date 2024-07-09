# alpha_max = 10
r_range = [ 0.3, 0.55, 0.8, 1.0 ]

# constants:
Sp = 4  # [ms]
Sd1, Sd2, Sd3 = 10, 15, 15  # [ms]
Sdk = 25  # [ms]

p0, p1, p2, p3, p4 = 0.15, 0.2, 0.15, 0.1, 0.4
p_same_disk_again = 0.25
p_sd_to_cpu = 0.35


for k in range(2, 6):
    for r in r_range:
        # u_cpu = 1 => lambda_0 = Sp => lambda_ = 0.64 * Sp
        alpha_max = min(
            0.64 * Sp,
            Sd1 / ((1 - p_same_disk_again) * p1),
            Sd2 / ((1 - p_same_disk_again) * p2),
            Sd3 / ((1 - p_same_disk_again) * p3),
            k * Sdk,
        )
        alpha_max -= 1e-11
        lambda_ = r * alpha_max

        lambda_0 = lambda_ / 0.64
        lambda_12 = (1 - p_same_disk_again) * (p1 * lambda_)
        lambda_22 = (1 - p_same_disk_again) * (p2 * lambda_)
        lambda_32 = (1 - p_same_disk_again) * (p3 * lambda_)
        lambda_di = lambda_ / k

        u_cpu = lambda_0 / Sp
        u_sd1 = lambda_12 / Sd1
        u_sd2 = lambda_22 / Sd2
        u_sd3 = lambda_32 / Sd3
        u_di = lambda_di / Sdk

        # rho == U
        rho_cpu = lambda_0 / Sp
        rho_sd1 = lambda_12 / Sd1
        rho_sd2 = lambda_22 / Sd2
        rho_sd3 = lambda_32 / Sd3
        rho_di = lambda_di / Sdk

        j_cpu = rho_cpu / (1 - rho_cpu)
        j_sd1 = rho_sd1 / (1 - rho_sd1)
        j_sd2 = rho_sd2 / (1 - rho_sd2)
        j_sd3 = rho_sd3 / (1 - rho_sd3)
        j_di = rho_di / (1 - rho_di)

        v_cpu = 1 / p4
        v_sd1 = p1 * v_cpu
        v_sd2 = p2 * v_cpu
        v_sd3 = p3 * v_cpu
        v_di = 1

        # following is same as r_i = 1 / (mu_i - lambda_i)
        # derived from R = J / X
        r_cpu = j_cpu / lambda_0
        r_sd1 = j_sd1 / lambda_12
        r_sd2 = j_sd2 / lambda_22
        r_sd3 = j_sd3 / lambda_32
        r_di = j_di / lambda_di

        # r_cpu = 1 / (Sp - lambda_0)
        # r_sd1 = 1 / (Sd1 - lambda_12)
        # r_sd2 = 1 / (Sd2 - lambda_22)
        # r_sd3 = 1 / (Sd3 - lambda_32)
        # r_di = 1 / (Sdk - lambda_di)

        R = v_cpu * r_cpu + v_sd1 * r_sd1 + v_sd2 * r_sd2 + v_sd3 *r_sd3 + v_di * r_di

        print(f'\n\nK: {k}, lambda: {lambda_} = {r} * {alpha_max}, r: {r}, alpha_max: {alpha_max}')

        print(f'Flow CPU: {lambda_0}')
        print(f'Flow Sd1: {lambda_12}')
        print(f'Flow Sd2: {lambda_22}')
        print(f'Flow Sd3: {lambda_32}')
        print(f'Flow Sdki: {lambda_di}')

        print(f'Usage CPU: {u_cpu}')
        print(f'Usage Sd1: {u_sd1}')
        print(f'Usage Sd2: {u_sd2}')
        print(f'Usage Sd3: {u_sd3}')
        print(f'Usage Sdki: {u_di}')

        print(f'Avg num of jobs on CPU: {j_cpu}')
        print(f'Avg num of jobs on Sd1: {j_sd1}')
        print(f'Avg num of jobs on Sd2: {j_sd2}')
        print(f'Avg num of jobs on Sd3: {j_sd3}')
        print(f'Avg num of jobs on Sdki: {j_di}')

        print(f'Avg time of a job\'s cycle: {R}')

        print(f'Intermediate result:')
        print(f'Avg num of visits in 1 cycle CPU: {v_cpu}')
        print(f'Avg num of visits in 1 cycle Sd1: {v_sd1}')
        print(f'Avg num of visits in 1 cycle Sd2: {v_sd2}')
        print(f'Avg num of visits in 1 cycle Sd3: {v_sd3}')
        print(f'Avg num of visits in 1 cycle Sdki: {v_di}')
        print(f'End of Intermediate result:')
