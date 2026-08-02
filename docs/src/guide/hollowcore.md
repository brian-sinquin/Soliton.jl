# Gas-Filled Hollow-Core PCF & Molecular Raman

GNLSE supports gas-filled Hollow-Core Photonic Crystal Fibers ([`HollowCoreFiber`](@ref)) and molecular gas Raman response models ([`MolecularRamanGas`](@ref)).

---

## ⚡ Physics Model

In hollow-core fibers (capillaries, Kagome, anti-resonant AR-PCF / Revolver fibers), light is guided inside a gas-filled central core. The propagation constant $\beta(\lambda, P)$ and non-linearity $\gamma(P)$ depend on gas pressure $P$ [bar] and core radius $R_{\text{core}}$:

$$\beta(\lambda, P) = \frac{2\pi}{\lambda} \sqrt{n_{\text{gas}}^2(\lambda, P) - \left( \frac{u_{01} \lambda}{2\pi R_{\text{core}}} \right)^2}$$

where $u_{01} \approx 2.40483$ is the fundamental $\text{HE}_{11}$ mode Bessel zero.

### Supported Gases

- **Noble Gases**: `:Ar` (Argon), `:Ne` (Neon), `:Kr` (Krypton), `:Xe` (Xenon).
- **Molecular Gases (Raman)**: `:H2` (Hydrogen), `:N2` (Nitrogen) — support both rotational and vibrational Raman lines.
- **Dispersion only**: `:Air` — Sellmeier dispersion available but no molecular Raman model.

---

## 💻 Usage Example

```julia
using GNLSE

grid = create_grid(2^13, 10e-12, 800e-9)
pulse = gaussian_pulse(grid, 5000.0, 50e-15)

# 30 μm core HC-PCF filled with 5 bar Argon at 800 nm
hcf = HollowCoreFiber(
    radius = 15e-6,      # 15 μm core radius (30 μm core diameter)
    gas = :Ar,           # Argon gas
    pressure = 5.0,      # 5 bar
    length = 0.3,        # 0.3 m propagation length
    lambda0 = 800e-9
)

params = SimParams(; medium=hcf, z_saves=100)
sol = solve(pulse, params)
```

By default `HollowCoreFiber` is lossless (`loss=0.0`). Pass `confinement_loss=true`
to additionally include the Marcatili-Schmeltzer capillary confinement loss
``\alpha(\lambda) \propto \lambda^2/a^3`` — note this bare-capillary formula is a
conservative bound that can overestimate loss for real anti-resonant/negative-curvature
HC-PCF designs (see `docs/src/physics.md`), so it defaults to off.

---

## 🧬 Molecular Gas Raman Response (`MolecularRamanGas`)

Molecular gases ($\text{H}_2, \text{N}_2$) exhibit narrow, high-frequency rotational and vibrational Raman transitions:

```julia
# Hydrogen rotational Raman model (17.6 THz shift)
h2_rot = MolecularRamanGas(:H2_rotational)

# Hydrogen vibrational Raman model (124.6 THz shift)
h2_vib = MolecularRamanGas(:H2_vibrational)

params = SimParams(; medium=hcf, raman_model=h2_rot, z_saves=100)
sol = solve(pulse, params)
```
