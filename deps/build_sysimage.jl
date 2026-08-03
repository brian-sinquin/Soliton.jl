using Pkg

# `create_sysimage(:Soliton)` needs Soliton to appear as a *dependency* of the
# active project. Simply activating the package's own repo root doesn't
# satisfy that — Soliton is that project, not a listed dependency of itself
# — so build in an isolated temp environment where it's `Pkg.develop`ed in.
pkg_root = dirname(@__DIR__)
tmp_env = mktempdir()
Pkg.activate(tmp_env)
Pkg.develop(; path=pkg_root)
Pkg.add("PackageCompiler")
Pkg.instantiate()

using PackageCompiler
using Libdl

println("=== Building Soliton Compiled Binary Sysimage ===")

sysimage_path = get(ENV, "SYSIMAGE_PATH", "Soliton_sysimage.$(Libdl.dlext)")
precompile_script = joinpath(pkg_root, "deps", "precompile_workload.jl")

create_sysimage(
    :Soliton;
    sysimage_path=sysimage_path,
    precompile_execution_file=precompile_script,
    incremental=true,
)

println("=== Successfully built Soliton sysimage: $sysimage_path ===")
