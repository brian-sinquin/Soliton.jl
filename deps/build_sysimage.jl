using Pkg

# Ensure PackageCompiler is available
Pkg.add("PackageCompiler")
using PackageCompiler
using Libdl

println("=== Building JuGNLSE Compiled Binary Sysimage ===")

sysimage_path = get(ENV, "SYSIMAGE_PATH", "JuGNLSE_sysimage.$(Libdl.dlext)")
precompile_script = joinpath(@__DIR__, "precompile_workload.jl")

create_sysimage(
    :JuGNLSE;
    sysimage_path=sysimage_path,
    precompile_execution_file=precompile_script,
    incremental=true,
)

println("=== Successfully built JuGNLSE sysimage: $sysimage_path ===")
