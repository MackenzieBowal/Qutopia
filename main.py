import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import json
from pulser_pasqal import PasqalCloud
import os
import pulser
def evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise
    distances are conserved"""
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    new_Q = squareform(
        DigitalAnalogDevice.interaction_coeff / pdist(new_coords) ** 6
    )
    return np.linalg.norm(new_Q - Q)

def run (input_data, solver_params,extra_arguments):
    
    Q = np.array(
        [
            [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
            [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
            [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
            [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
            [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
        ]
    )
    bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
    costs = []
    # this takes exponential time with the dimension of the QUBO
    for b in bitstrings:
        z = np.array(list(b), dtype=int)
        cost = z.T @ Q @ z
        costs.append(cost)
    zipped = zip(bitstrings, costs)
    sort_zipped = sorted(zipped, key=lambda x: x[1])
    shape = (len(Q), 2)
    costs = []
    np.random.seed(0)
    x0 = np.random.random(shape).flatten()
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q, shape),
        method="Nelder-Mead",
        tol=1e-6,
        options={"maxiter": 200000, "maxfev": None},
    )
    coords = np.reshape(res.x, (len(Q), 2))
    qubits = dict(enumerate(coords))
    reg = Register(qubits)
    
    # We choose a median value between the min and the max
    Omega = np.median(Q[Q > 0].flatten())
    delta_0 = -5  # just has to be negative
    delta_f = -delta_0  # just has to be positive
    T = 4000  # time in ns, we choose a time long enough to ensure the propagation of information in the system
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    seq = Sequence(reg,  pulser.MockDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")
    ############################ FOR LOCAL SIMULATION ############################
    simul = QutipEmulator.from_sequence(seq)
    results = simul.run()
    ##############################################################################
    ########### THE CURRENT SOLVER IS CREATED FOR ONLY LOCAL SIMULATION ##########     
    ### PLEASE, VISIT THE EXAMPLES AND THE DOCUMENTATION FOR REMOTE SIMULATION ###
    ##############################################################################
    
    ########################### FOR REMOTE SIMULATION ############################   
    connection = PasqalCloud(
        username=os.environ.get('PASQAL_USERNAME'),  # Your username or email address for the Pasqal Cloud Platform
        project_id=os.environ.get('PASQAL_PROJECTID'),  # The ID of the project associated to your account
        password=os.environ.get('PASQAL_PASSWORD'),  # The password for your Pasqal Cloud Platform account
    )
    emu_tn_default = pulser.backends.EmuTNBackend.default_config
    simul = pulser.backends.EmuTNBackend(
        seq, connection=connection, config=emu_tn_default
    )
    # Remote execution, requires job_params
    job_params = [
        {"runs": 100, "variables": {"t": 1000}},
        {"runs": 50, "variables": {"t": 2000}},
    ]
    #results = simul.run(job_params=job_params)
    ##############################################################################

    final = results.get_final_state()
    count_dict = results.sample_final_state()
    return {x:int(count_dict[x]) for x in count_dict}
