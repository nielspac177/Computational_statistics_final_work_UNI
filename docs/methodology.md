# Statistical Methodology

## Overview

This project implements a comprehensive statistical analysis comparing classical Poisson regression with random effects extensions for modeling over-dispersed count data. The methodology combines maximum likelihood estimation, Monte Carlo hypothesis testing, and simulation-based validation.

## 1. Classical Poisson Regression

### Model Specification
For count data $Y_i$ representing disease cases in district $i$:

$$Y_i \sim \text{Poisson}(\lambda_i = N_i R_i)$$
$$\log(R_i) = \beta_0 + x_i \beta_1$$

Where:
- $N_i$: Population of district $i$
- $R_i$: Incidence rate in district $i$  
- $x_i$: Predictor variable (socioeconomic index)
- $\beta_0, \beta_1$: Regression coefficients

### Likelihood Function
The log-likelihood (excluding constants):

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i (\log(N_i) + \beta_0 + x_i \beta_1) - N_i \exp(\beta_0 + x_i \beta_1) \right]$$

### Estimation
- **Method**: Newton-Raphson optimization with automatic differentiation
- **Implementation**: `Optim.jl` with `Newton()` algorithm  
- **Starting values**: $\beta_0 = 0, \beta_1 = 0$

## 2. Over-dispersion Testing

### Test Statistic
The Pearson chi-squared statistic:

$$T = \sum_{i=1}^n \frac{(Y_i - \hat{\lambda}_i)^2}{\hat{\lambda}_i}$$

Under $H_0$ (equi-dispersion): $T \sim \chi^2_{n-p}$ asymptotically.

### Monte Carlo Approach
For enhanced accuracy:
1. Simulate $Y_i^{(m)} \sim \text{Poisson}(\hat{\lambda}_i)$ for $m = 1, \ldots, M$
2. Compute $T^{(m)}$ for each simulation
3. Calculate empirical p-value: $p = \frac{1}{M} \sum_{m=1}^M \mathbf{1}(T^{(m)} > T_{\text{obs}})$

**Implementation Details:**
- $M = 2000$ Monte Carlo replications
- Random seed fixed for reproducibility
- Exact finite-sample distribution

## 3. Random Effects Model

### Hierarchical Specification
$$Y_i | Z_i \sim \text{Poisson}(\lambda_i = N_i R_i)$$
$$\log(R_i) = \beta_0 + x_i \beta_1 + Z_i$$
$$Z_i \sim \mathcal{N}(0, \sigma_z^2)$$

### Marginal Properties
The random effects induce over-dispersion:
- **Mean**: $\mathbb{E}[Y_i] = N_i \exp(\beta_0 + x_i \beta_1 + \sigma_z^2/2)$
- **Variance**: $\text{Var}[Y_i] > \mathbb{E}[Y_i]$ when $\sigma_z^2 > 0$

### Likelihood Evaluation
The marginal likelihood requires numerical integration:

$$L(\boldsymbol{\beta}, \sigma_z^2) = \prod_{i=1}^n \int_{-\infty}^{\infty} \frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!} \cdot \frac{1}{\sqrt{2\pi\sigma_z^2}} \exp\left(-\frac{z^2}{2\sigma_z^2}\right) dz$$

**Numerical Integration:**
- **Method**: Adaptive Gaussian quadrature (`QuadGK.jl`)
- **Integration bounds**: $[-5\sigma_z, 5\sigma_z]$ (99.9% coverage)
- **Tolerance**: Relative error $< 10^{-6}$

### Parameter Estimation
- **Optimization**: Nelder-Mead algorithm (derivative-free)
- **Parameterization**: $\theta = (\beta_0, \beta_1, \log \sigma_z)$ ensures $\sigma_z > 0$
- **Starting values**: Classical Poisson estimates + $\sigma_z = 0.1$

## 4. Model Comparison

### Information Criteria
Akaike Information Criterion:
$$\text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2p$$

**Model Selection:**
- Lower AIC indicates better fit
- $\Delta\text{AIC} > 2$: substantial evidence
- $\Delta\text{AIC} > 10$: strong evidence

### Likelihood Ratio Considerations
While a formal LRT is complex due to boundary conditions ($\sigma_z^2 = 0$), the enormous AIC difference provides overwhelming evidence for the random effects model.

## 5. Bootstrap Inference

### Non-parametric Bootstrap
For classical Poisson model confidence intervals:

1. **Resample**: Draw $n$ districts with replacement
2. **Refit**: Estimate parameters on bootstrap sample  
3. **Repeat**: $B = 500$ bootstrap replications
4. **Intervals**: Use empirical quantiles for $(1-\alpha) \times 100\%$ CI

**Advantages:**
- Distribution-free
- Accounts for finite-sample effects
- Robust to model misspecification

## 6. Simulation Study

### Validation Framework
**Objective**: Verify parameter recovery under known conditions

**Design:**
- **True parameters**: $\beta_0 = -3.0, \beta_1 = 0.5, \sigma_z = 0.3$
- **Data structure**: Use real population sizes and predictor values
- **Replications**: $S = 50$ independent datasets
- **Sample size**: $n = 40$ districts

**Data Generation:**
1. Generate $Z_i \sim \mathcal{N}(0, \sigma_z^2)$
2. Compute $\lambda_i = N_i \exp(\beta_0 + x_i \beta_1 + Z_i)$
3. Simulate $Y_i \sim \text{Poisson}(\lambda_i)$

### Performance Metrics
- **Bias**: $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$
- **RMSE**: $\text{RMSE}(\hat{\theta}) = \sqrt{\mathbb{E}[(\hat{\theta} - \theta)^2]}$
- **Success Rate**: Proportion of convergent optimizations

## 7. Implementation Details

### Computational Considerations
- **Numerical stability**: Log-space computations avoid overflow
- **Convergence criteria**: Default tolerances in `Optim.jl`
- **Error handling**: Graceful failure for non-convergent cases
- **Reproducibility**: Fixed random seeds throughout

### Software Architecture
```julia
# Main analysis pipeline
1. Data loading and exploration
2. Classical Poisson fitting  
3. Over-dispersion testing
4. Random effects fitting
5. Model comparison
6. Bootstrap analysis
7. Simulation validation
8. Results summarization
```

### Performance Characteristics
- **Runtime**: 2-3 minutes for complete analysis
- **Memory**: ~2GB peak usage during simulation
- **Scalability**: Efficient for datasets up to ~1000 observations

## 8. Statistical Assumptions

### Classical Poisson
1. **Independence**: Observations are independent
2. **Equi-dispersion**: $\mathbb{E}[Y_i] = \text{Var}[Y_i]$
3. **Log-linear mean**: Exponential link function appropriate

### Random Effects Extension
1. **Independence**: Conditional on random effects
2. **Normality**: Random effects are Gaussian
3. **Exchangeability**: Random effects identically distributed

### Robustness
- Over-dispersion test robust to moderate assumption violations
- Bootstrap inference distribution-free
- Random effects accommodate various sources of extra variation

## 9. Interpretation Guidelines

### Parameter Interpretation
- **$\beta_1$**: Log rate ratio per unit increase in predictor
- **$\exp(\beta_1)$**: Multiplicative effect on incidence rate  
- **$\sigma_z$**: Standard deviation of unobserved heterogeneity

### Model Assessment
- **AIC differences**: Quantify relative model support
- **Residual analysis**: Check for systematic patterns
- **Simulation validation**: Confirms methodology soundness

This comprehensive methodology ensures robust statistical inference while maintaining computational efficiency and interpretability.
