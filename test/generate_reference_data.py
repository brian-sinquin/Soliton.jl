"""
generate_reference_data.py
==========================
Generates CSV reference data for GNLSE adversarial comparison tests using
gnlse-python 2.0.0 (https://pypi.org/project/gnlse/).

Run once to regenerate fixtures (committed to the repo as golden data):
    C:\\Espressif\\tools\\python\\python.exe test/generate_reference_data.py

Output files written to: test/reference_data/
  gnlse_soliton.csv       - Fundamental soliton: peak power vs z
  gnlse_spm.csv           - Pure SPM: RMS spectral width vs z
  gnlse_dispersion.csv    - Linear dispersion: pulse FWHM vs z
  gnlse_raman_ssfs.csv    - Soliton + BlowWood Raman: spectral centroid vs z
  gnlse_field_output.csv  - Full field at fiber output (soliton scenario)
  gnlse_tpa.csv           - TPA via complex gamma: peak power vs z
  gnlse_tpa_field.csv     - Full field at waveguide output (TPA scenario)

Physical quantities are stored in SI units (m, s, W, rad/s) even though
gnlse-python works internally in ps/nm. Conversion is applied on output.

gnlse-python conventions (v2.0.0):
  - time_window: float [ps]
  - wavelength:  float [nm]
  - GNLSESetup.W: self-steepening ratio W/w0 (not the full W grid)
  - Solution.t:   time axis [ps]
  - Solution.Omega: angular frequency axis [rad/ps]
  - Solution.At:  time-domain field [sqrt(W)], shape (z_saves, N)
  - Solution.AW:  freq-domain field [sqrt(W)·ps], shape (z_saves, N)
"""

import numpy as np
import gnlse
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "reference_data")
os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────
c = 299792458.0  # m/s

# ─────────────────────────────────────────────
# Shared grid parameters (in gnlse-python units)
# ─────────────────────────────────────────────
N         = 2**12
T_WIN_PS  = 20.0       # ps
LAM_NM    = 835.0      # nm
BETA2_SI  = -1.0e-26   # s^2/m  (anomalous)
GAMMA_SI  = 0.1        # 1/(W·m)

# gnlse-python uses [1/W/m] for nonlinearity directly
# but betas in [ps^n / m] with n starting at 2
BETA2_PS2M = BETA2_SI * 1e24  # s^2/m → ps^2/m:  1 s^2 = 1e24 ps^2

def _rms_width(spectrum, omega_axis):
    """RMS spectral width from |A(omega)|^2 spectrum on omega axis."""
    P = np.abs(spectrum)**2
    m = np.sum(omega_axis * P) / np.sum(P)
    return np.sqrt(np.sum((omega_axis - m)**2 * P) / np.sum(P))

def _fwhm(intensity, axis):
    """Intensity FWHM on a monotonic axis."""
    half = 0.5 * np.max(intensity)
    above = np.where(intensity >= half)[0]
    if len(above) < 2:
        return 0.0
    return axis[above[-1]] - axis[above[0]]

def _spectral_centroid(spectrum, omega_axis):
    P = np.abs(spectrum)**2
    return np.sum(omega_axis * P) / np.sum(P)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: Fundamental soliton — peak power should stay constant vs z
# ─────────────────────────────────────────────────────────────────────────────
print("Running Scenario 1: Fundamental soliton...")

T0_SOL_PS = 0.1  # ps = 100 fs 1/e half-width
P0_SOL    = abs(BETA2_SI) / (GAMMA_SI * (T0_SOL_PS * 1e-12)**2)  # W

setup = gnlse.GNLSESetup()
setup.resolution      = N
setup.time_window     = T_WIN_PS          # ps
setup.wavelength      = LAM_NM            # nm
setup.fiber_length    = 0.4              # m
setup.z_saves         = 21
setup.nonlinearity    = GAMMA_SI          # 1/W/m
setup.raman_model     = None
setup.self_steepning  = False
setup.dispersion_model = gnlse.DispersionFiberFromTaylor(
    BETA2_PS2M, [BETA2_PS2M]
)
# Sech pulse: intensity FWHM = 2*arcsinh(1)*T0 ≈ 1.7627*T0
fwhm_ps = 2 * np.log(1 + np.sqrt(2)) * T0_SOL_PS
setup.pulse_model = gnlse.SechEnvelope(P0_SOL, fwhm_ps)
setup.rtol = 1e-8
setup.atol = 1e-10

solver = gnlse.GNLSE(setup)
sol = solver.run()

z_m = sol.Z                              # [m]
peak_power = np.max(np.abs(sol.At)**2, axis=1)  # W

header = "z_m,peak_power_W"
data = np.column_stack([z_m, peak_power])
np.savetxt(os.path.join(OUTDIR, "gnlse_soliton.csv"), data,
           delimiter=",", header=header, comments="")
print(f"  P0 = {P0_SOL:.3f} W, peak drift = {(peak_power[-1]-peak_power[0])/peak_power[0]*100:.3f}%")

# Also save full output field (last z slice)
At_out = sol.At[-1, :]                  # [sqrt(W)]
t_s    = sol.t * 1e-12                  # ps → s
header2 = "t_s,At_real_sqrtW,At_imag_sqrtW"
data2 = np.column_stack([t_s, At_out.real, At_out.imag])
np.savetxt(os.path.join(OUTDIR, "gnlse_field_output.csv"), data2,
           delimiter=",", header=header2, comments="")
print("  Saved gnlse_soliton.csv and gnlse_field_output.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Pure SPM — RMS spectral width should broaden monotonically
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Scenario 2: Pure SPM spectral broadening...")

setup_spm = gnlse.GNLSESetup()
setup_spm.resolution      = N
setup_spm.time_window     = T_WIN_PS
setup_spm.wavelength      = LAM_NM
setup_spm.fiber_length    = 0.5          # m
setup_spm.z_saves         = 21
setup_spm.nonlinearity    = GAMMA_SI
setup_spm.raman_model     = None
setup_spm.self_steenpening = False
setup_spm.dispersion_model = None        # dispersionless
setup_spm.pulse_model      = gnlse.SechEnvelope(100.0, 0.1)  # 100 W, 100 fs
setup_spm.rtol = 1e-8
setup_spm.atol = 1e-10

solver_spm = gnlse.GNLSE(setup_spm)
sol_spm = solver_spm.run()

rms_widths = np.array([
    _rms_width(sol_spm.AW[i, :], sol_spm.W * 1e12)  # rad/ps → rad/s
    for i in range(len(sol_spm.Z))
])

data_spm = np.column_stack([sol_spm.Z, rms_widths])
np.savetxt(os.path.join(OUTDIR, "gnlse_spm.csv"), data_spm,
           delimiter=",",
           header="z_m,rms_spectral_width_rad_s",
           comments="")
print(f"  Initial RMS width: {rms_widths[0]:.4e} rad/s")
print(f"  Final   RMS width: {rms_widths[-1]:.4e} rad/s")
print("  Saved gnlse_spm.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Linear dispersion — Gaussian FWHM broadens as sqrt(1+(L/LD)^2)
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Scenario 3: Linear dispersion broadening...")

FWHM_G_PS = 0.2    # 200 fs Gaussian (intensity FWHM)
T0G_PS    = FWHM_G_PS / (2 * np.sqrt(np.log(2)))
LD        = (T0G_PS * 1e-12)**2 / abs(BETA2_SI)  # m

setup_disp = gnlse.GNLSESetup()
setup_disp.resolution      = N
setup_disp.time_window     = T_WIN_PS
setup_disp.wavelength      = LAM_NM
setup_disp.fiber_length    = LD           # propagate exactly one L_D
setup_disp.z_saves         = 21
setup_disp.nonlinearity    = 0.0          # linear only
setup_disp.raman_model     = None
setup_disp.self_steenpening = False
setup_disp.dispersion_model = gnlse.DispersionFiberFromTaylor(
    BETA2_PS2M, [BETA2_PS2M]
)
setup_disp.pulse_model = gnlse.GaussianEnvelope(1.0, FWHM_G_PS)
setup_disp.rtol = 1e-8
setup_disp.atol = 1e-10

solver_disp = gnlse.GNLSE(setup_disp)
sol_disp = solver_disp.run()

t_s_disp = sol_disp.t * 1e-12   # s

fwhm_vals = np.array([
    _fwhm(np.abs(sol_disp.At[i, :])**2, t_s_disp)
    for i in range(len(sol_disp.Z))
])

data_disp = np.column_stack([sol_disp.Z, fwhm_vals])
np.savetxt(os.path.join(OUTDIR, "gnlse_dispersion.csv"), data_disp,
           delimiter=",",
           header="z_m,temporal_fwhm_s",
           comments="")
print(f"  L_D = {LD:.4f} m")
print(f"  Initial FWHM: {fwhm_vals[0]*1e15:.1f} fs")
print(f"  Final   FWHM: {fwhm_vals[-1]*1e15:.1f} fs  (theory: {FWHM_G_PS*np.sqrt(2)*1e3:.1f} fs)")
print("  Saved gnlse_dispersion.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Soliton + Raman (BlowWood) — spectral centroid must red-shift
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Scenario 4: Raman self-frequency shift...")

T0_RAMAN_PS = 0.05   # 50 fs
P0_RAMAN    = abs(BETA2_SI) / (GAMMA_SI * (T0_RAMAN_PS * 1e-12)**2)

setup_raman = gnlse.GNLSESetup()
setup_raman.resolution      = N
setup_raman.time_window     = T_WIN_PS
setup_raman.wavelength      = LAM_NM
setup_raman.fiber_length    = 0.4
setup_raman.z_saves         = 21
setup_raman.nonlinearity    = GAMMA_SI
setup_raman.raman_model     = gnlse.raman_blowwood
setup_raman.self_steenpening = False
setup_raman.dispersion_model = gnlse.DispersionFiberFromTaylor(
    BETA2_PS2M, [BETA2_PS2M]
)
fwhm_raman_ps = 2 * np.log(1 + np.sqrt(2)) * T0_RAMAN_PS
setup_raman.pulse_model = gnlse.SechEnvelope(P0_RAMAN, fwhm_raman_ps)
setup_raman.rtol = 1e-8
setup_raman.atol = 1e-10

solver_raman = gnlse.GNLSE(setup_raman)
sol_raman = solver_raman.run()

omega_abs = sol_raman.W * 1e12  # sol_raman.W is absolute frequency in rad/ps -> rad/s

centroid_abs = np.array([
    _spectral_centroid(sol_raman.AW[i, :], omega_abs)
    for i in range(len(sol_raman.Z))
])

data_raman = np.column_stack([sol_raman.Z, centroid_abs])
np.savetxt(os.path.join(OUTDIR, "gnlse_raman_ssfs.csv"), data_raman,
           delimiter=",",
           header="z_m,spectral_centroid_rad_s",
           comments="")
print(f"  Initial centroid: {centroid_abs[0]:.6e} rad/s")
print(f"  Final   centroid: {centroid_abs[-1]:.6e} rad/s")
total_shift_nm = (2 * np.pi * c / centroid_abs[-1] - 2 * np.pi * c / centroid_abs[0]) * 1e9
print(f"  SSFS wavelength shift: {total_shift_nm:.2f} nm (should be > 0, red-shift)")
print("  Saved gnlse_raman_ssfs.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Scenario 7: Two-Photon Absorption via the "complex gamma" trick
#
# gnlse-python has no dedicated TPA/free-carrier model, but its nonlinear step
# is `rv = 1j * gamma * W * M * exp(-D*z)` with M = ifft(A * |A|^2) (Kerr-only,
# no Raman here). Passing a *complex* `setup.nonlinearity = gamma_r + 1j*gamma_i`
# gives `1j*gamma_r*M - gamma_i*M`, i.e. Kerr phase (Im part) plus a `-gamma_i*|A|^2*A`
# loss term -- exactly the field-domain TPA equation
#   dA/dz|_TPA = -(alpha2/(2*Aeff)) * |A|^2 * A
# with gamma_i = alpha2/(2*Aeff). This is the standard silicon-photonics
# "complex gamma = Kerr + i*TPA" convention (Lin, Painter & Agrawal, Opt.
# Express 15, 16604 (2007)), so it lets the *existing* gnlse-python dependency
# serve as an independent, external solver (scipy solve_ivp RK45, not Soliton's
# own ERK4IP/SSFM) to cross-validate the TPA channel -- unlike 3PA and FCA/FCR,
# which need an auxiliary carrier-density ODE / >cubic field nonlinearity that
# gnlse-python's Kerr-only nonlinear step cannot express.
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Scenario 7: Two-photon absorption via complex gamma...")

ALPHA2_SI = 5.0e-12    # m/W  (TPA coefficient, ~0.5 cm/GW Si @ 1550 nm)
AEFF_SI   = 0.1e-12    # m^2
GAMMA_TPA_SI = 100.0   # 1/(W.m)  (Kerr part)
BETAS_TPA_SI = [-1000e-27]  # s^2/m
LAM_TPA_NM = 1550.0
L_TPA = 0.005          # m
P0_TPA = 50.0          # W
FWHM_TPA_PS = 1.0      # ps (Gaussian)

gamma_i = ALPHA2_SI / (2.0 * AEFF_SI)          # 1/(W.m), TPA loss rate
gamma_complex = GAMMA_TPA_SI + 1j * gamma_i    # standard Kerr + i*TPA convention

betas_tpa_ps2m = [b * 1e24 for b in BETAS_TPA_SI]  # s^n/m -> ps^n/m

setup_tpa = gnlse.GNLSESetup()
setup_tpa.resolution      = N
setup_tpa.time_window     = T_WIN_PS
setup_tpa.wavelength      = LAM_TPA_NM
setup_tpa.fiber_length    = L_TPA
setup_tpa.z_saves         = 21
setup_tpa.nonlinearity    = gamma_complex      # complex! see note above
setup_tpa.raman_model     = None
setup_tpa.self_steenpening = False
setup_tpa.dispersion_model = gnlse.DispersionFiberFromTaylor(
    0.0, betas_tpa_ps2m   # loss=0 (isolate TPA-only loss, no extra linear loss)
)
setup_tpa.pulse_model = gnlse.GaussianEnvelope(P0_TPA, FWHM_TPA_PS)
setup_tpa.rtol = 1e-10
setup_tpa.atol = 1e-12

solver_tpa = gnlse.GNLSE(setup_tpa)
sol_tpa = solver_tpa.run()

z_tpa = sol_tpa.Z
peak_tpa = np.max(np.abs(sol_tpa.At)**2, axis=1)

data_tpa = np.column_stack([z_tpa, peak_tpa])
np.savetxt(os.path.join(OUTDIR, "gnlse_tpa.csv"), data_tpa,
           delimiter=",", header="z_m,peak_power_W", comments="")

At_tpa_out = sol_tpa.At[-1, :]
t_tpa_s = sol_tpa.t * 1e-12
data_tpa_field = np.column_stack([t_tpa_s, At_tpa_out.real, At_tpa_out.imag])
np.savetxt(os.path.join(OUTDIR, "gnlse_tpa_field.csv"), data_tpa_field,
           delimiter=",", header="t_s,At_real_sqrtW,At_imag_sqrtW", comments="")

print(f"  P0 = {P0_TPA} W, alpha2 = {ALPHA2_SI:.2e} m/W, Aeff = {AEFF_SI:.2e} m^2")
print(f"  Peak power in:  {peak_tpa[0]:.4f} W")
print(f"  Peak power out: {peak_tpa[-1]:.4f} W  (transmission {100*peak_tpa[-1]/peak_tpa[0]:.1f}%)")
print("  Saved gnlse_tpa.csv and gnlse_tpa_field.csv")


print("\n✅ All reference data generated successfully.")
print(f"   Output directory: {OUTDIR}")
