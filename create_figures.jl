# Create figures for the LaTeX report
# Run this after the main analysis to generate publication-ready figures

using CairoMakie
using Printf

# Set publication theme
set_theme!(theme_minimal())
update_theme!(
    fontsize = 12,
    Axis = (
        titlesize = 14,
        labelsize = 12,
        ticklabelsize = 10
    )
)

println("Creating figures for LaTeX report...")

# Ensure we have the data and results from main analysis
if !@isdefined(covid_data)
    println("Running main analysis first...")
    include("poisson_overdispersion_analysis.jl")
end

# Create figures directory
mkpath("figures")

# Figure 1: Overdispersion Test
println("Creating Figure 1: Overdispersion test...")
fig1 = Figure(resolution = (600, 400))
ax1 = Axis(fig1[1, 1], 
    title = "Monte Carlo Test for Over-dispersion",
    xlabel = "Test Statistic",
    ylabel = "Density",
    titlesize = 14
)

hist!(ax1, sim_stats, bins=50, normalization=:pdf, 
      color=(:steelblue, 0.6), strokewidth=1, strokecolor=:steelblue)
vlines!(ax1, [obs_stat], color=:red, linewidth=3, 
        label="Observed = $(round(obs_stat, digits=1))")
axislegend(ax1, position=:rt)

save("figures/figure1_overdispersion_test.png", fig1, px_per_unit=2)
save("figures/figure1_overdispersion_test.pdf", fig1)

# Figure 2: Model Comparison - Observed vs Predicted
println("Creating Figure 2: Model comparison...")
λ_classical = predict_poisson(β_poisson, covid_data)
λ_random_effects = covid_data.population .* exp.(β_pn[1] .+ covid_data.predictor * β_pn[2])

fig2 = Figure(resolution = (800, 400))

# Classical Poisson
ax2a = Axis(fig2[1, 1], 
    title = "Classical Poisson",
    xlabel = "Predicted Cases",
    ylabel = "Observed Cases",
    aspect = 1
)
scatter!(ax2a, λ_classical, covid_data.cases, markersize=8, color=:steelblue)
ablines!(ax2a, 0, 1, color=:red, linestyle=:dash, linewidth=2)

# Random effects model
ax2b = Axis(fig2[1, 2], 
    title = "Poisson with Random Effects",
    xlabel = "Predicted Cases",
    ylabel = "Observed Cases",
    aspect = 1
)
scatter!(ax2b, λ_random_effects, covid_data.cases, markersize=8, color=:darkorange)
ablines!(ax2b, 0, 1, color=:red, linestyle=:dash, linewidth=2)

save("figures/figure2_model_comparison.png", fig2, px_per_unit=2)
save("figures/figure2_model_comparison.pdf", fig2)

# Figure 3: Bootstrap Distributions
println("Creating Figure 3: Bootstrap distributions...")
fig3 = Figure(resolution = (800, 400))

ax3a = Axis(fig3[1, 1], 
    title = "Bootstrap Distribution of β₀",
    xlabel = "β₀",
    ylabel = "Density"
)
hist!(ax3a, bootstrap_results[:, 1], bins=30, normalization=:pdf, 
      color=(:steelblue, 0.6), strokewidth=1)
vlines!(ax3a, [β_poisson[1]], color=:red, linewidth=2, label="Estimate")
vlines!(ax3a, ci_β₀, color=:orange, linewidth=2, linestyle=:dash, label="95% CI")
axislegend(ax3a, position=:rt)

ax3b = Axis(fig3[1, 2], 
    title = "Bootstrap Distribution of β₁",
    xlabel = "β₁",
    ylabel = "Density"
)
hist!(ax3b, bootstrap_results[:, 2], bins=30, normalization=:pdf, 
      color=(:steelblue, 0.6), strokewidth=1)
vlines!(ax3b, [β_poisson[2]], color=:red, linewidth=2, label="Estimate")
vlines!(ax3b, ci_β₁, color=:orange, linewidth=2, linestyle=:dash, label="95% CI")
axislegend(ax3b, position=:rt)

save("figures/figure3_bootstrap_distributions.png", fig3, px_per_unit=2)
save("figures/figure3_bootstrap_distributions.pdf", fig3)

# Figure 4: Simulation Results
if @isdefined(estimates_clean) && size(estimates_clean, 1) > 0
    println("Creating Figure 4: Simulation results...")
    fig4 = Figure(resolution = (900, 300))
    
    true_values = [true_β₀, true_β₁, true_σ_z]
    param_names = ["β₀", "β₁", "σ_z"]
    
    for i in 1:3
        ax = Axis(fig4[1, i], 
            title = "$(param_names[i]) Estimates",
            xlabel = param_names[i],
            ylabel = "Density"
        )
        hist!(ax, estimates_clean[:, i], bins=20, normalization=:pdf, 
              color=(:steelblue, 0.6), strokewidth=1)
        vlines!(ax, [true_values[i]], color=:red, linewidth=3, label="True")
        vlines!(ax, [mean(estimates_clean[:, i])], color=:orange, linewidth=2, label="Mean")
        if i == 3
            axislegend(ax, position=:rt)
        end
    end
    
    save("figures/figure4_simulation_results.png", fig4, px_per_unit=2)
    save("figures/figure4_simulation_results.pdf", fig4)
end

# Create a summary table
println("Creating results table...")
open("figures/results_table.tex", "w") do f
    write(f, """
\\begin{table}[H]
\\centering
\\begin{tabular}{lccc}
\\toprule
Model & Log-likelihood & AIC & Parameters \\\\
\\midrule
Classical Poisson & $(round(loglik_classical, digits=1)) & $(round(aic_classical, digits=1)) & 2 \\\\
Random Effects & $(round(loglik_random_effects, digits=1)) & $(round(aic_random_effects, digits=1)) & 3 \\\\
\\midrule
\\multicolumn{2}{l}{ΔAIC (improvement)} & $(round(aic_classical - aic_random_effects, digits=1)) & \\\\
\\bottomrule
\\end{tabular}
\\caption{Model comparison results}
\\label{tab:model_comparison}
\\end{table}
""")
end

println("All figures created successfully in the 'figures/' directory!")
println("Files created:")
println("- figure1_overdispersion_test.pdf")
println("- figure2_model_comparison.pdf") 
println("- figure3_bootstrap_distributions.pdf")
if @isdefined(estimates_clean)
    println("- figure4_simulation_results.pdf")
end
println("- results_table.tex")
println("\nAdd these to your LaTeX document!")
