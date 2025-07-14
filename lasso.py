# 🎯 Key Recommendations:
# Start with Minimal Space:

# 3 parameters: alpha, fit_intercept, positive
# 9 combinations to explore efficiently
# Most bang for your buck

# Alpha Range Adjustment:

# Your 1e-8 to 100.0 is too wide
# 1e-5 to 1.0 covers most practical cases
# 1e-6 to 10.0 if you want to be thorough

# Why These Parameters Matter:

# alpha: Controls sparsity vs. fit trade-off (most important)
# fit_intercept: Whether to center data (dataset-dependent)
# positive: Useful for interpretability in some domains
# selection: Can affect convergence speed
# -------------------------------------------------------------
# 1. precompute

# Usually auto-handled by sklearn
# Manual tuning rarely improves performance
# Can slow down optimization

# 2. copy_X

# Memory management parameter, not performance-related
# Should be True by default to avoid data corruption
# No need to tune

# 3. warm_start

# Only useful for iterative fitting with changing parameters
# Not beneficial in cross-validation context
# Can cause issues

# 4. max_iter range too wide

# 500-20000 is excessive for most datasets
# Lasso usually converges much faster
# Wide range wastes trials