# COMMON CONFIG

K_RANGE = [ i for i in range(2, 6) ]

R_RANGE = [ 0.3, 0.55, 0.8, 1.0 ]

# RESOURCES_ALIASES[0] = 'Generator', etc.
RESOURCES_ALIASES = [ 'Generator', 'CPU', 'Sd1', 'Sd2', 'Sd3' ]

PROBS = [
    # generator cpu   Sd1   Sd2   Sd3
    [ 0.,       1,    0,    0,    0],
    [ 0.,       0.15, 0.2,  0.15, 0.1, ],
    [ 0.,       0.35, 0.25, 0,    0,   ],
    [ 0.,       0.35, 0,    0.25, 0,   ],
    [ 0.,       0.35, 0,    0,    0.25,],
]

#         cpu   Sd1   Sd2   Sd3  Di
SERVICE_TIMES = [ 4.,  10,   15,   15,  25 ]



# ANALYTICAL SOLUTION CONFIG:


PATH_REL_FLOWS_RESULTS_ANALYTICALLY = r'./throughputs_analytical.csv'

PATH_ALL_RESULTS_ANALYTICALLY = r'./results_analytical.csv'

PATH_ALPHA_MAX_PLOT = r'./alpha_max.png'


# SIMULATED SOLUTION CONFIG:


PATH_SIMULATION_RESULTS = r'results_simulation.csv'

PATH_SIMULATIONS_AVG_RESULTS = r'results_simulations_avg.csv'

SIMULATION_AVG_NUM_RERUNS = 2

SIMULATION_DURATION = 30  # [mins]
SIMULATION_DURATION = 1  # [mins]
