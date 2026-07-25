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
            "Getting Started"          => "guide/basic.md",
            "Dispersion Models"        => "guide/dispersion.md",
            "Raman Scattering"         => "guide/raman.md",
            "Nonlinearity Models"      => "guide/nonlinearity.md",
            "Cascaded Propagation"     => "guide/cascading.md",
            "Birefringent Propagation" => "guide/vectorial.md",
            "Commercial Fiber Catalog" => "guide/fibers.md",
            "Amplifying Fibers (EDFA)" => "guide/edfa.md",
            "Noise Modeling"           => "guide/noise.md",
        ],
        "Examples" => [
            "Overview"                    => "examples/index.md",
            "1 — Supercontinuum in PCF"    => "examples/ex1_supercontinuum.md",
            "2 — Soliton Self-Freq. Shift" => "examples/ex2_ssfs.md",
            "3 — SC Coherence"            => "examples/ex3_coherence.md",
            "4 — Soliton Trapping"        => "examples/ex4_birefringence.md",
            "5 — HOSoliton Compression"   => "examples/ex5_soliton_compression.md",
            "6 — Stable N=3 Soliton"      => "examples/ex6_stable_n3_soliton.md",
        ],
        "API Reference" => [
            "Medium"           => "api/medium.md",
            "Grid"             => "api/grid.md",
            "Pulses"           => "api/pulse.md",
            "Solvers"          => "api/solvers.md",
            "Dispersion"       => "api/dispersion.md",
            "Nonlinearity"     => "api/nonlinearity.md",
            "Lumped Elements"  => "api/elements.md",
            "Fiber Catalog"    => "api/fibers.md",
            "Analysis"         => "api/analysis.md",
        ],
    ],
    warnonly=true,
)

deploydocs(; repo="github.com/brian-sinquin/JuGNLSE.jl", devbranch="master")
