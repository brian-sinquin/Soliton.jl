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
    size --> (1000, 700)
    titlefontsize --> 11

    t_ps = sol.t .* 1e12
    Z_m = sol.Z

    # `sol.W` (absolute angular frequency ω₀+V) can include non-physical W <= 0
    # bins at the edges of an oversampled grid (e.g. a wide-bandwidth grid around
    # a near-infrared pump). Drop those before converting to wavelength, since
    # λ = 2πc/ω blows up (or flips sign) near ω = 0 and would otherwise dominate
    # the auto-ranged wavelength axis and hide the real spectrum.
    valid = findall(>(0), sol.W)
    lambda_nm_raw = (2π * c ./ sol.W[valid]) .* 1e9
    ord = valid[sortperm(lambda_nm_raw)]
    lambda_nm = (2π * c ./ sol.W[ord]) .* 1e9

    P_t = abs2.(sol.At)
    P_w = abs2.(sol.AW[ord, :])

    # dB scale (standard for GNLSE/supercontinuum dynamics, which routinely span
    # several orders of magnitude in power) with a floor to keep colors readable.
    floor_db = -40.0
    Pt_max = maximum(P_t)
    Pw_max = maximum(P_w)
    Pt_db = clamp.(10 .* log10.(P_t ./ Pt_max .+ eps()), floor_db, 0.0)
    Pw_db_raw = 10 .* log10.(P_w ./ Pw_max .+ eps())   # unclamped: used to find the significant band
    Pw_db = clamp.(Pw_db_raw, floor_db, 0.0)           # clamped: used for display

    # Crop to the energetically significant band (bins whose peak-over-z power is
    # within `floor_db` of the global peak) instead of the full raw grid extent.
    # This must slice the actual plotted arrays (not just set `xlims`) because GR
    # resamples heatmaps over the full data domain of non-uniformly-spaced x data,
    # and a domain spanning to the near-zero-frequency edge (λ → huge) can blow up
    # its internal grid allocation.
    peak_per_bin = vec(maximum(Pw_db_raw; dims=2))
    significant = findall(>=(floor_db), peak_per_bin)
    lo_idx, hi_idx = isempty(significant) ? (1, length(lambda_nm)) :
                     (first(significant), last(significant))
    lambda_nm = lambda_nm[lo_idx:hi_idx]
    Pw_db = Pw_db[lo_idx:hi_idx, :]
    lam_lo, lam_hi = extrema(lambda_nm)

    # Subplot 1: Temporal Evolution Heatmap
    @series begin
        subplot := 1
        seriestype := :heatmap
        xlabel := "Time t [ps]"
        ylabel := "Distance z [m]"
        title := "Temporal Evolution [dB]"
        color := :turbo
        clims := (floor_db, 0.0)
        colorbar := true
        t_ps, Z_m, Pt_db'
    end

    # Subplot 2: Spectral Evolution Heatmap
    @series begin
        subplot := 2
        seriestype := :heatmap
        xlabel := "Wavelength λ [nm]"
        ylabel := "Distance z [m]"
        title := "Spectral Evolution [dB]"
        color := :turbo
        clims := (floor_db, 0.0)
        colorbar := true
        xlims := (lam_lo, lam_hi)
        lambda_nm, Z_m, Pw_db'
    end

    # Subplot 3: Initial vs Final Time Profile
    @series begin
        subplot := 3
        seriestype := :path
        xlabel := "Time t [ps]"
        ylabel := "Power |A(t)|² [W]"
        title := "Temporal Slice (z=0 vs z=L)"
        label := "z = 0"
        t_ps, P_t[:, 1]
    end

    @series begin
        subplot := 3
        seriestype := :path
        label := "z = L"
        t_ps, P_t[:, end]
    end

    # Subplot 4: Initial vs Final Spectrum Profile
    @series begin
        subplot := 4
        seriestype := :path
        xlabel := "Wavelength λ [nm]"
        ylabel := "Power [dB]"
        title := "Spectral Slice (z=0 vs z=L)"
        xlims := (lam_lo, lam_hi)
        ylims := (floor_db, 0.0)
        label := "z = 0"
        lambda_nm, Pw_db[:, 1]
    end

    @series begin
        subplot := 4
        seriestype := :path
        xlims := (lam_lo, lam_hi)
        ylims := (floor_db, 0.0)
        label := "z = L"
        lambda_nm, Pw_db[:, end]
    end
end
