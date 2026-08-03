# 🚀 Soliton Benchmark Results Summary

| Benchmark Target | Median Time | Memory | Allocations | Comparison |
| :--- | :--- | :--- | :--- | :--- |
| `solvers/ssfm_vectorial` | 74.25 ms | 9.18 MiB | 22472 | Baseline |
| `solvers/ssfm_scalar` | 49.30 ms | 4.71 MiB | 186 | Baseline |
| `solvers/erk4ip_scalar` | 226.49 ms | 5.15 MiB | 207 | Baseline |
| `operators/vectorial_spm_fwm_eval` | 40.20 μs | 1.59 KiB | 22 | Baseline |
| `operators/build_physics_model` | 113.70 μs | 1.08 MiB | 86 | Baseline |
| `operators/spm_raman_eval` | 33.90 μs | 0 B | 0 | Baseline |
| `analysis/spectral_coherence` | 49.00 μs | 128.32 KiB | 10 | Baseline |
| `analysis/pulse_energy` | 695.19 ns | 48 B | 3 | Baseline |
| `analysis/add_noise` | 447.80 μs | 1.02 MiB | 36383 | Baseline |
