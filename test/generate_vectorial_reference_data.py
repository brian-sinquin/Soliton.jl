"""
Reference data generator for Vectorial / Birefringent Coupled GNLSE.

Uses `cnlse` (https://github.com/WUST-FOG/cgnlse-python), the WUST-FOG group's
published Coupled Generalized NLSE solver built directly on top of the real
`gnlse` package (the same package used for the scalar adversarial scenarios).
This is a peer-reviewed third-party implementation, not a hand-rolled script.

Run once to regenerate the fixture (committed to the repo as golden data):
    C:\\Espressif\\tools\\python\\python.exe test/generate_vectorial_reference_data.py

Requires: pip install gnlse==2.0.0 scipy tqdm
The `cnlse` package source is vendored locally in test/cnlse_pkg/ (pulled from
the cgnlse-python GitHub repo; not published on PyPI).

Physics/convention notes
-------------------------
`cnlse.CNLSE` follows the exact same FFT convention as `gnlse-python` and
GNLSE (`AW = ifft(At)`, `At = fft(AW)`), so no sign flips are needed for
beta1 here (unlike a naive hand-written SSFM using the opposite convention).

`CNLSE`'s coupled nonlinear term only includes the coherent FWM/XPM term
(the (1/3)*Ay^2*conj(Ax)*exp(-2i*deltaBeta*z) term) when a Raman model is
supplied — the no-Raman branch silently drops it. GNLSE's vectorial
solver (`src/nonlinearity.jl::_vectorial_spm_fwm`) always includes it, matching
the full coupled-NLS equations for a linearly birefringent fiber (Agrawal,
"Nonlinear Fiber Optics", Ch. 6). To get an apples-to-apples comparison, we
pass a Raman model with fr=0 (zero response), which triggers CNLSE's
Raman-branch formula (including the coherent term) while contributing zero
actual Raman effect.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cnlse_pkg"))

import gnlse
from cnlse import CNLSE, DubbleDispersionFiberFromTaylor


def raman_zero(T):
    """Zero-strength Raman model: activates CNLSE's coherent-FWM branch
    without contributing any actual delayed Raman response."""
    z = np.zeros_like(T)
    return 0.0, z, z, z


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    os.makedirs(data_dir, exist_ok=True)

    print("Generating Vectorial / Birefringent Coupled GNLSE Reference Data (cnlse package)...")

    N = 2048
    T_WIN_PS = 20.0          # ps
    LAM_NM = 1550.0          # nm
    L = 1.0                  # m
    GAMMA = 0.0012           # 1/W/m (matches GNLSE BirefringentMedium gamma)

    # beta2_x = beta2_y = -21.5e-27 s^2/m -> ps^2/m
    BETA2_PS2M = -21.5e-27 * 1e24  # = -0.0215 ps^2/m

    # GNLSE: TaylorDispersion(betas, beta1) with beta1_x=+0.5e-12 s/m, beta1_y=-0.5e-12 s/m.
    # DubbleDispersionFiberFromTaylor defines:
    #   Lx = -1j*deltaB1/2*V + ...   (so beta1_x = -deltaB1/2)
    #   Ly = +1j*deltaB1/2*V + ...   (so beta1_y = +deltaB1/2)
    # => deltaB1 = -2 * beta1_x_ps_per_m
    BETA1_X_PS_M = 0.5e-12 * 1e12   # = 0.5 ps/m
    DELTA_B1 = -2.0 * BETA1_X_PS_M  # ps/m

    setup = gnlse.GNLSESetup()
    setup.resolution = N
    setup.time_window = T_WIN_PS
    setup.wavelength = LAM_NM
    setup.fiber_length = L
    setup.z_saves = 2
    setup.nonlinearity = [GAMMA, GAMMA]
    setup.raman_model = raman_zero          # activates coherent-FWM branch, fr=0
    setup.self_steepning = False
    # deltaN = 0 -> deltaB (phase-mismatch term) = 0, matching GNLSE's deltabeta0=0.0
    setup.dispersion_model = DubbleDispersionFiberFromTaylor(
        0.0, np.array([[BETA2_PS2M], [BETA2_PS2M]]), 0.0, DELTA_B1, 2 * np.pi * gnlse.common.c / LAM_NM
    )
    setup.rtol = 1e-8
    setup.atol = 1e-10

    t = np.linspace(-T_WIN_PS / 2, T_WIN_PS / 2, N)
    m = 2 * np.log(1 + np.sqrt(2))
    FWHM_PS = 1.0e-12 * 1e12  # 1 ps
    P0 = 500.0
    Ax0 = np.sqrt(P0) * 2 / (np.exp(m * t / FWHM_PS) + np.exp(-m * t / FWHM_PS))
    Ay0 = Ax0.copy()
    setup.pulse_model = np.concatenate((Ax0, Ay0)).astype(complex)

    solver = CNLSE(setup)
    solution = solver.run()

    # solution.At has shape (2*z_saves, N): first z_saves rows = x, next = y
    Atx_out = solution.At[setup.z_saves - 1, :]
    Aty_out = solution.At[2 * setup.z_saves - 1, :]

    t_s = t * 1e-12  # ps -> s

    out_file = os.path.join(data_dir, "vectorial_soliton_trapping.csv")
    data = np.column_stack((
        t_s,
        np.real(Atx_out), np.imag(Atx_out), np.abs(Atx_out) ** 2,
        np.real(Aty_out), np.imag(Aty_out), np.abs(Aty_out) ** 2,
    ))
    header = "time_s,Ax_real,Ax_imag,Ax_intensity,Ay_real,Ay_imag,Ay_intensity"
    np.savetxt(out_file, data, delimiter=",", header=header, comments="")
    print(f"Saved: {out_file}")
    print(f"  Peak |Ax|^2 = {np.max(np.abs(Atx_out)**2):.4f} W, Peak |Ay|^2 = {np.max(np.abs(Aty_out)**2):.4f} W")


if __name__ == "__main__":
    main()
