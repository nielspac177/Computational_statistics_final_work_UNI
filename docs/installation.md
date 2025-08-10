# Installation Guide

## System Requirements

- **Julia**: Version 1.11.6 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: At least 4GB RAM recommended
- **Storage**: ~500MB for Julia packages

## Step 1: Install Julia

### macOS (recommended)
```bash
brew install julia
```

### Alternative: Download from Official Site
1. Go to [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Download Julia 1.11.6+ for your system
3. Follow installation instructions

## Step 2: Install Required Julia Packages

### Quick Setup
```bash
julia test_environment.jl
```

### Manual Installation
```julia
using Pkg

# Core packages
Pkg.add(["CSV", "DataFrames", "Distributions", "Random"])
Pkg.add(["LinearAlgebra", "Optim", "StatsBase", "StatsFuns"])
Pkg.add(["SpecialFunctions", "QuadGK", "Printf"])

# Visualization
Pkg.add("CairoMakie")

# Jupyter support (optional)
Pkg.add("IJulia")
```

## Step 3: Verify Installation

Run the environment test:
```bash
julia test_environment.jl
```

Expected output:
```
✓ CSV - OK
✓ DataFrames - OK
✓ Distributions - OK
✓ Random - OK
✓ LinearAlgebra - OK
✓ Optim - OK
✓ StatsBase - OK
✓ StatsFuns - OK
✓ QuadGK - OK
✓ Printf - OK

✓ All required packages are available!
```

## Step 4: Set Up Jupyter (Optional)

For notebook support:
```julia
using IJulia
IJulia.installkernel("Julia")
```

Then install Jupyter:
```bash
pip install jupyter notebook
```

## Troubleshooting

### Package Installation Issues
If you encounter package installation problems:

```julia
using Pkg
Pkg.update()
Pkg.resolve()
```

### Julia Kernel Not Found in Jupyter
```julia
using IJulia
IJulia.installkernel("Julia", "--project=@.")
```

### Permission Issues (macOS/Linux)
```bash
sudo chown -R $(whoami) ~/.julia
```

### Optimization Convergence Issues
If the random effects model fails to converge:
- Increase the number of iterations in Optim settings
- Try different starting values
- Check data for extreme outliers

## Performance Optimization

### For Faster Execution
```julia
# Use multiple threads
export JULIA_NUM_THREADS=4

# Precompile packages
julia -e 'using Pkg; Pkg.precompile()'
```

### Memory Considerations
- The analysis requires ~2GB RAM during execution
- Simulation studies may need more memory for large replications
- Consider reducing simulation size if memory is limited

## Package Versions

Tested with:
```
Julia 1.11.6
CSV v0.10.15
DataFrames v1.7.0
Distributions v0.25.120
Optim v1.13.2
CairoMakie v0.15.5
QuadGK v2.11.2
StatsBase v0.34.6
StatsFuns v1.5.0
SpecialFunctions v2.5.1
```

## Getting Help

If you encounter issues:

1. **Check package status**: `julia -e 'using Pkg; Pkg.status()'`
2. **Update packages**: `julia -e 'using Pkg; Pkg.update()'`
3. **Clear package cache**: Remove `~/.julia/compiled` directory
4. **Reinstall Julia**: Download fresh version from julialang.org

## Next Steps

After successful installation:
1. Run `julia poisson_overdispersion_analysis.jl` for complete analysis
2. Open `poisson_overdispersion_analysis.ipynb` in Jupyter for interactive analysis
3. Check `figures/` directory for generated visualizations
