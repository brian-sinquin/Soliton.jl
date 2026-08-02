using Test
using GNLSE

@testset verbose = true "GNLSE.jl" begin
    include("test_unit.jl")
    include("test_api.jl")
    include("test_solvers.jl")
    include("test_physics.jl")
    include("test_vectorial.jl")
    include("test_fibers.jl")
    include("test_edfa.jl")
    include("test_hollowcore.jl")
    include("test_semiconductor.jl")
    include("test_loss_gain.jl")
    include("test_conversions.jl")
    include("test_adversarial.jl")
    include("test_recipes.jl")
end
