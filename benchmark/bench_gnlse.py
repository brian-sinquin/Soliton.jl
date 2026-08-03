import gnlse
import numpy as np
import time

# 1. Pulse parameters
N = 2**14
time_window = 10.0 # ps
lambda0 = 835.0 # nm
T0 = 0.05 # ps
P0 = 10000.0 # W

# Setup grid and envelope
setup = gnlse.SolitonSetup()
setup.resolution = N
setup.time_window = time_window
setup.wavelength = lambda0
setup.fiber_length = 0.15 # m
setup.raman_model = gnlse.raman_blowwood
setup.self_steepening = True

# Betas in ps^n / m (convert from s^n / m by multiplying by (1e12)^n)
betas_si = [-11.830e-27, 8.1038e-41, -9.5205e-56, 2.0737e-70,
            -5.3943e-85, 1.3486e-99, -2.5495e-114, 3.0524e-129,
            -1.7140e-144]
betas = [b * (1e12)**(i+2) for i, b in enumerate(betas_si)]
# gnlse-python uses Taylor expansion
dispersion = gnlse.DispersionFiberFromTaylor(
    loss=0,
    betas=betas
)
setup.dispersion_model = dispersion
setup.nonlinearity = 0.11

# Pulse
setup.pulse_model = gnlse.envelopes.SechEnvelope(Pmax=P0, FWHM=T0 * 1.763)

# Model
solver = gnlse.Soliton(setup)

print("Running gnlse-python benchmark...")
start_time = time.time()
# gnlse uses run()
solution = solver.run()
end_time = time.time()
exec_time = end_time - start_time

print(f"Execution time: {exec_time} s")

with open("time_gnlse.txt", "w") as f:
    f.write(str(exec_time) + "\n")
