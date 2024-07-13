# COMMON CONFIG

K_RANGE = [ i for i in range(0, 6) ]

R_RANGE = [ 0.3, 0.55, 0.8, 1.0 ]

# names instead of their indices in PROBS and SERVICE_TIMES_MS
RESOURCES_ALIASES = [ 'Generator', 'CPU', 'Sd1', 'Sd2', 'Sd3' ]

PROBS = [
    # generator cpu   Sd1   Sd2   Sd3
    [ 0.,       1,    0,    0,    0    ],
    [ 0.,       0.15, 0.2,  0.15, 0.1, ],
    [ 0.,       0.35, 0.25, 0,    0,   ],
    [ 0.,       0.35, 0,    0.25, 0,   ],
    [ 0.,       0.35, 0,    0,    0.25,],
]

#                   cpu   Sd1   Sd2   Sd3  Di
SERVICE_TIMES_MS = [ 4,  10,   15,   15,  25 ]  # [ms]



# ANALYTICAL SOLUTION CONFIG:


PATH_REL_FLOWS_RESULTS_ANALYTICALLY = r'./results/relative_throughputs_analytically.csv'

PATH_ALL_RESULTS_ANALYTICALLY = r'./results/stats_analytically.csv'

PATH_ALPHA_MAX_PLOT = r'./results/alpha_maxs.png'



# SIMULATED SOLUTION CONFIG:


PATH_SIMULATION_RESULTS = r'./results/stats_simulation.csv'

PATH_SIMULATIONS_AVG_RESULTS = r'./results/stats_simulations_avg.csv'

SIMULATION_AVG_NUM_RERUNS = 2

# SIMULATION_DURATION_MINS = 30  # [mins]
SIMULATION_DURATION_MINS = 1  # [mins]
