using RecipesBase

"""
    @recipe function f(sol::Solution)

Plot recipe for `Solution` creating a 4-panel dashboard:
1. Temporal evolution heat map |A(z, t)|²
2. Spectral evolution heat map |A(z, λ)|² (or |A(z, ω)|²)
3. Initial vs Final Temporal Intensity
4. Initial vs Final Spectral Power
"""
@recipe function f(sol::Solution)
    layout := (2, 2)
    legend := false

    t_ps = sol.t .* 1e12
    Z_m = sol.Z
    lambda_nm = (2π * c ./ sol.W) .* 1e9

    # Subplot 1: Temporal Evolution Heatmap
    @series begin
        subplot := 1
        seriestype := :heatmap
        xlabel := "Time t [ps]"
        ylabel := "Distance z [m]"
        title := "Temporal Intensity Evolution"
        t_ps, Z_m, abs2.(sol.At)'
    end

    # Subplot 2: Spectral Evolution Heatmap
    @series begin
        subplot := 2
        seriestype := :heatmap
        xlabel := "Wavelength λ [nm]"
        ylabel := "Distance z [m]"
        title := "Spectral Power Evolution"
        lambda_nm, Z_m, abs2.(sol.AW)'
    end

    # Subplot 3: Initial vs Final Time Profile
    @series begin
        subplot := 3
        seriestype := :path
        xlabel := "Time t [ps]"
        ylabel := "Power |A(t)|² [W]"
        title := "Temporal Slice (z=0 vs z=L)"
        label := "z = 0"
        t_ps, abs2.(sol.At[:, 1])
    end

    @series begin
        subplot := 3
        seriestype := :path
        label := "z = L"
        t_ps, abs2.(sol.At[:, end])
    end

    # Subplot 4: Initial vs Final Spectrum Profile
    @series begin
        subplot := 4
        seriestype := :path
        xlabel := "Wavelength λ [nm]"
        ylabel := "Power |A(λ)|² [W·s²]"
        title := "Spectral Slice (z=0 vs z=L)"
        label := "z = 0"
        lambda_nm, abs2.(sol.AW[:, 1])
    end

    @series begin
        subplot := 4
        seriestype := :path
        label := "z = L"
        lambda_nm, abs2.(sol.AW[:, end])
    end
end
