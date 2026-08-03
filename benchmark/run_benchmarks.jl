using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using BenchmarkTools
using JSON
using Printf

include("benchmarks.jl")

println("=========================================================")
println("  Soliton Performance Benchmark Suite")
println("=========================================================")

# Tune and run benchmark suite
println("\n[1/3] Tuning benchmarks...")
tune!(SUITE; verbose=false)

println("[2/3] Running benchmark suite...")
results = run(SUITE; verbose=true)

# Convert results to clean serializable dictionary
function benchmark_to_dict(b)
    m = median(b)
    return Dict("time_ns" => m.time, "memory_bytes" => m.memory, "allocs" => m.allocs)
end

function group_to_dict(g)
    d = Dict{String, Any}()
    for (k, v) in g
        if v isa BenchmarkGroup
            d[k] = group_to_dict(v)
        else
            d[k] = benchmark_to_dict(v)
        end
    end
    return d
end

results_dict = group_to_dict(results)

# Helper formatting functions
function format_time(ns)
    if ns < 1e3
        return @sprintf("%.2f ns", ns)
    elseif ns < 1e6
        return @sprintf("%.2f μs", ns / 1e3)
    elseif ns < 1e9
        return @sprintf("%.2f ms", ns / 1e6)
    else
        return @sprintf("%.3f s", ns / 1e9)
    end
end

function format_bytes(b)
    if b < 1024
        return "$b B"
    elseif b < 1024^2
        return @sprintf("%.2f KiB", b / 1024)
    else
        return @sprintf("%.2f MiB", b / 1024^2)
    end
end

# Check for previous baseline
baseline_path = joinpath(@__DIR__, "baseline.json")
baseline_dict = isfile(baseline_path) ? JSON.parsefile(baseline_path) : nothing

# Generate Markdown Summary
println("\n[3/3] Generating summary report...")

summary_io = IOBuffer()
println(summary_io, "# 🚀 Soliton Benchmark Results Summary\n")
println(
    summary_io, "| Benchmark Target | Median Time | Memory | Allocations | Comparison |"
)
println(summary_io, "| :--- | :--- | :--- | :--- | :--- |")

function process_group_markdown(group_name, group_data, baseline_group=nothing)
    for (name, data) in group_data
        t_str = format_time(data["time_ns"])
        m_str = format_bytes(data["memory_bytes"])
        a_str = "$(data["allocs"])"

        comp_str = "Baseline"
        if !isnothing(baseline_group) && haskey(baseline_group, name)
            b_data = baseline_group[name]
            ratio = data["time_ns"] / b_data["time_ns"]
            pct = (1.0 - ratio) * 100.0
            if pct > 1.0
                comp_str = @sprintf("🟢 **%.1f%% faster**", pct)
            elseif pct < -1.0
                comp_str = @sprintf("🔴 **%.1f%% slower**", -pct)
            else
                comp_str = "⚪ ~Same"
            end
        end

        println(
            summary_io, "| `$group_name/$name` | $t_str | $m_str | $a_str | $comp_str |"
        )
    end
end

for (group_name, group_data) in results_dict
    base_grp = if !isnothing(baseline_dict) && haskey(baseline_dict, group_name)
        baseline_dict[group_name]
    else
        nothing
    end
    process_group_markdown(group_name, group_data, base_grp)
end

summary_md = String(take!(summary_io))
println("\n" * summary_md)

# Write summary and results to disk
write(joinpath(@__DIR__, "summary.md"), summary_md)
write(joinpath(@__DIR__, "results.json"), JSON.json(results_dict, 2))

# Save current results as baseline if baseline doesn't exist
if !isfile(baseline_path)
    write(baseline_path, JSON.json(results_dict, 2))
    println("\nSaved current benchmark run as initial `baseline.json`.")
end

println("Benchmark execution completed successfully.")
