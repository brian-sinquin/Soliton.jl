import laserfun as lf
import numpy as np
import time

# 1. Pulse parameters
N = 2**14
time_window = 10.0  # 10 ps
lambda0 = 835.0 # nm
T0 = 0.05 # ps
P0 = 10000.0

# Create pulse
pulse = lf.Pulse(pulse_type='sech', power=P0, fwhm_ps=T0 * 1.763, center_wavelength_nm=lambda0,
                 time_window_ps=time_window, npts=N)

# 2. Fiber parameters
# Betas in ps^n / m (convert from s^n / m by multiplying by (1e12)^n)
betas_si = [-11.830e-27, 8.1038e-41, -9.5205e-56, 2.0737e-70,
            -5.3943e-85, 1.3486e-99, -2.5495e-114, 3.0524e-129,
            -1.7140e-144]
betas = [b * (1e12)**(i+2) for i, b in enumerate(betas_si)]

# Laserfun betas are given directly in an array starting from beta2
fiber1 = lf.Fiber(length=0.15, center_wl_nm=lambda0, dispersion=betas,
                  gamma_W_m=0.11)

# 3. Solver Setup
# laserfun uses an adaptive solver by default (RK45)
# If we want a fixed step size we can use steps=1500 but adaptive is its default.
# For fairness with Julia, let's use adaptive or fixed step depending on the comparison.
# We will use adaptive since it's the default in laserfun.
# We'll just run it.

print("Running laserfun benchmark...")
start_time = time.time()
results = lf.NLSE(pulse, fiber1, raman=True, custom_raman='dudley', shock=True, print_status=False)
end_time = time.time()
exec_time = end_time - start_time

print(f"Execution time: {exec_time} s")

with open("time_laserfun.txt", "w") as f:
    f.write(str(exec_time) + "\n")
