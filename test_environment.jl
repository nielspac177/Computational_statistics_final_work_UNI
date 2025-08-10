# Test Julia Environment for Poisson Analysis
# Run this script first to check if all packages are available

println("Testing Julia environment for Poisson analysis...")

# Test package availability
required_packages = [
    "CSV", "DataFrames", "Distributions", "Random", 
    "LinearAlgebra", "Optim", "StatsBase", "StatsFuns", 
    "QuadGK", "Printf"
]

missing_packages = String[]

for pkg in required_packages
    try
        eval(Meta.parse("using $pkg"))
        println("✓ $pkg - OK")
    catch e
        println("✗ $pkg - MISSING")
        push!(missing_packages, pkg)
    end
end

if !isempty(missing_packages)
    println("\nMissing packages detected. Please install:")
    println("using Pkg")
    for pkg in missing_packages
        println("Pkg.add(\"$pkg\")")
    end
else
    println("\n✓ All required packages are available!")
    
    # Test basic functionality
    println("\nTesting basic functionality...")
    
    # Test CSV reading
    try
        test_data = DataFrame(x = [1, 2, 3], y = [4, 5, 6])
        println("✓ DataFrame creation - OK")
    catch e
        println("✗ DataFrame creation failed")
    end
    
    # Test optimization
    try
        f(x) = (x[1] - 1)^2 + (x[2] - 2)^2
        result = optimize(f, [0.0, 0.0])
        println("✓ Optimization - OK")
    catch e
        println("✗ Optimization failed")
    end
    
    # Test distributions
    try
        d = Poisson(5.0)
        x = rand(d, 10)
        println("✓ Random number generation - OK")
    catch e
        println("✗ Random number generation failed")
    end
    
    # Test numerical integration
    try
        result, _ = quadgk(x -> x^2, 0, 1)
        println("✓ Numerical integration - OK")
    catch e
        println("✗ Numerical integration failed")
    end
    
    println("\n✓ Environment test completed successfully!")
    println("You can now run the main analysis: include(\"poisson_overdispersion_analysis.jl\")")
end
