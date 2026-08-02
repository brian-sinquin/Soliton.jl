using JuGNLSE
using Documenter

DocMeta.setdocmeta!(JuGNLSE, :DocTestSetup, :(using JuGNLSE); recursive=true)

makedocs(;
    modules=[JuGNLSE],
    authors="Brian Sinquin <148503669+brian-sinquin@users.noreply.github.com> and contributors",
    sitename="JuGNLSE.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://brian-sinquin.github.io/JuGNLSE.jl",
        edit_link="master",
        assets=String[],
        mathengine=Documenter.MathJax3(),
    ),
    pages=[
        "Home" => "index.md",
        "Physics Background" => "physics.md",
        "User Guides" => [
            "Getting Started" => "guide/basic.md",
            "Dispersion Models" => "guide/dispersion.md",
            "Raman Scattering" => "guide/raman.md",
            "Nonlinearity Models" => "guide/nonlinearity.md",
            "Cascaded Propagation" => "guide/cascading.md",
            "Birefringent Propagation" => "guide/vectorial.md",
            "Commercial Fiber Catalog" => "guide/fibers.md",
            "Amplifying Fibers (EDFA)" => "guide/edfa.md",
            "Hollow-Core PCF" => "guide/hollowcore.md",
            "Silicon & Semiconductors" => "guide/semiconductor.md",
            "Noise Modeling" => "guide/noise.md",
        ],
        "Examples" => [
            "Overview" => "examples/index.md",
            "1 — Supercontinuum in PCF" => "examples/ex1_supercontinuum.md",
            "2 — Soliton Self-Freq. Shift" => "examples/ex2_ssfs.md",
            "3 — SC Coherence" => "examples/ex3_coherence.md",
            "4 — Soliton Trapping" => "examples/ex4_birefringence.md",
            "5 — HOSoliton Compression" => "examples/ex5_soliton_compression.md",
            "6 — Stable N=3 Soliton" => "examples/ex6_stable_n3_soliton.md",
            "7 — Hollow-Core Gas Fiber" => "examples/ex7_hollowcore_gas.md",
            "8 — Silicon Photonics (TPA)" => "examples/ex8_silicon_tpa.md",
            "9 — EDFA Pulse Amplifier" => "examples/ex9_edfa_amplifier.md",
            "10 — Multithreaded Sweep" => "examples/ex10_parallel_sweep.md",
        ],
        "API Reference" => [
            "Medium" => "api/medium.md",
            "Grid" => "api/grid.md",
            "Pulses" => "api/pulse.md",
            "Solvers" => "api/solvers.md",
            "Dispersion" => "api/dispersion.md",
            "Nonlinearity" => "api/nonlinearity.md",
            "Lumped Elements" => "api/elements.md",
            "Fiber Catalog" => "api/fibers.md",
            "Analysis" => "api/analysis.md",
            "Unit Conversions" => "api/conversions.md",
        ],
    ],
    # Keep style/completeness issues non-fatal, but let `@example`/`@eval`/`@setup`
    # block execution failures fail the build — a silently-broken example (missing
    # output, no plot) is worse than a red CI check. `:missing_docs` is no longer
    # suppressed: every exported symbol now has an `@docs` entry, so a new export
    # without documentation should fail CI rather than silently vanish from the
    # API reference.
    warnonly=[
        :autodocs_block,
        :cross_references,
        :docs_block,
        :footnote,
        :linkcheck_remotes,
        :linkcheck,
        :meta_block,
    ],
)

deploydocs(;
    repo="github.com/brian-sinquin/JuGNLSE.jl", devbranch="master", push_preview=true
)
