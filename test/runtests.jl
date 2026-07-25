using Test
using JuGNLSE

@testset verbose = true "JuGNLSE.jl" begin
    include("test_unit.jl")
    include("test_api.jl")
    include("test_solvers.jl")
    include("test_physics.jl")
    include("test_vectorial.jl")
    include("test_fibers.jl")
end
