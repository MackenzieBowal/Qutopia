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

B = 6.56
# B = 9.62

def run (input_data, solver_params,extra_arguments):
    coords = input_data["points"]
    rB = input_data["bird_radius"]

    print(f"Bird radius: {rB}")

    Omega = 2 * np.pi
    delta_0 = -5  # just has to be negative
    delta_f = -delta_0  # just has to be positive
    T = 16000  # time in ns, we choose a time long enough to ensure the propagation of information in the system
    
    B = DigitalAnalogDevice.rydberg_blockade_radius(Omega)

    print(B)
    
    qubits = dict(enumerate([[v / rB * B for v in c] for c in coords]))
    reg = Register(qubits)
    
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    seq = Sequence(reg,  pulser.MockDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")
    ############################ FOR LOCAL SIMULATION ############################
    simul = QutipEmulator.from_sequence(seq, sampling_rate=0.1)

    print("starting sim")
    
    results = simul.run(progress_bar=True)

    print("finished sim")
    
    ##############################################################################
    ########### THE CURRENT SOLVER IS CREATED FOR ONLY LOCAL SIMULATION ##########     
    ### PLEASE, VISIT THE EXAMPLES AND THE DOCUMENTATION FOR REMOTE SIMULATION ###
    ##############################################################################
    
    ########################### FOR REMOTE SIMULATION ############################   
    # connection = PasqalCloud(
    #     username=os.environ.get('PASQAL_USERNAME'),  # Your username or email address for the Pasqal Cloud Platform
    #     project_id=os.environ.get('PASQAL_PROJECTID'),  # The ID of the project associated to your account
    #     password=os.environ.get('PASQAL_PASSWORD'),  # The password for your Pasqal Cloud Platform account
    # )
    # emu_tn_default = pulser.backends.EmuTNBackend.default_config
    # simul = pulser.backends.EmuTNBackend(
    #     seq, connection=connection, config=emu_tn_default
    # )
    # # Remote execution, requires job_params
    # job_params = [
    #     {"runs": 100, "variables": {"t": 1000}},
    #     {"runs": 50, "variables": {"t": 2000}},
    # ]
    #results = simul.run(job_params=job_params)
    ##############################################################################

    final = results.get_final_state()
    count_dict = results.sample_final_state()
    return reg, {x:int(count_dict[x]) for x in count_dict}
