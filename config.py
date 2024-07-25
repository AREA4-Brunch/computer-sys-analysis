# ===================================================
# COMMON CONFIG


K_RANGE = [ i for i in range(0, 6) ]  # num of usr disks to add

R_RANGE = [ 0.3, 0.55, 0.8, 1.0 ]  # generator's alpha's scaling factor

# use names in results instead of their indices in PROBS and SERVICE_TIMES_MS
RESOURCES_ALIASES = [ 'Generator', 'CPU', 'Sd1', 'Sd2', 'Sd3' ]

# PROBS[src][dst] = prob of task completed on src going to dst
#                   if row does not sum up to 1 rest is prob of task leaving net
PROBS = [
    # generator cpu   Sd1   Sd2   Sd3
    [ 0.,       1,    0,    0,    0    ],
    [ 0.,       0.15, 0.2,  0.15, 0.1, ],
    [ 0.,       0.35, 0.25, 0,    0,   ],
    [ 0.,       0.35, 0,    0.25, 0,   ],
    [ 0.,       0.35, 0,    0,    0.25,],
]

# service time of generator (alpha) is not a param (is computed dynamically)
#                   cpu   Sd1   Sd2   Sd3  Di
SERVICE_TIMES_MS = [ 4,  10,   15,   15,  25 ]  # [ms]



# ===================================================
# ANALYTICAL SOLUTION CONFIG:


PATH_REL_FLOWS_RESULTS_ANALYTICALLY = r'./results/relative_throughputs_analytically.csv'

PATH_ALL_RESULTS_ANALYTICALLY = r'./results/stats_analytically.csv'

PATH_ALPHA_MAX_PLOT = r'./results/alpha_maxs.png'



# ===================================================
# SIMULATED SOLUTION CONFIG:


PATH_SIMULATION_RESULTS = r'./results/stats_simulation.csv'  # single run results

PATH_SIMULATIONS_AVG_RESULTS = r'./results/stats_simulations_avg.csv'

SIMULATION_AVG_NUM_RERUNS = 2  # avg out results over some num of reruns

SIMULATION_DURATION_MINS = 1  # [mins] - mins of simulated time
