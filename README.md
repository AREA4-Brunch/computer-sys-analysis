It started as coursework, but I turned the simulation part into a design patterns exercise.

Network can be provided in the `config.py`.
User disks can be added to the network in the config file.
Service time of the generator (`alpha`) is computed as `alpha_max * scaling_factor`, where `alpha_max` is
analytically computed value of `alpha` at which the network becomes saturated.
Scaling factor values can be provided in the config file.
<br/>
Results are computed for all combinations of given numbers of user disks and scaling factors.
<br/>


`solve_analytically.py`
- provides estimates for usages, throughputs and num jobs on each resource
<br/>

`solve_simulated.py`
- simulates the network and calculates usages, throughputs and num jobs on each resource, as well as the job generator
- after a first run it starts parallel reruns and averages their results out
