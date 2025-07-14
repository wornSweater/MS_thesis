def create_optimized_study(storage_url=None):
    """Highly optimized Optuna setup with best practices"""
    
    # Advanced TPE sampler configuration
    sampler = TPESampler(
        n_startup_trials=20,      # More random exploration
        n_ei_candidates=50,       # More candidates for better selection
        gamma=0.25,              # Top 25% trials for modeling
        prior_weight=1.0,        # Weight of prior distribution
        seed=42,
        multivariate=True,       # Enable multivariate TPE
        group=True,             # Enable categorical grouping
        warn_independent_sampling=True
    )
    
    # Advanced pruning with Hyperband
    pruner = HyperbandPruner(
        min_resource=1,          # Minimum resource allocation
        max_resource=100,        # Maximum resource allocation  
        reduction_factor=3,      # Resource reduction factor
        bootstrap_count=10       # Number of bootstrap samples
    )
    
    # Create study with optional distributed storage
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,     # For distributed optimization
        study_name="optimized_study",
        load_if_exists=True      # Resume if study exists
    )
    
    return study

def optimized_optimize(study, objective_func, n_trials=200):
    """Run optimization with best practices configuration"""
    
    # Pre-warm with known good parameters if available
    # study.enqueue_trial({"param1": 0.1, "param2": 10})
    
    study.optimize(
        objective_func,
        n_trials=n_trials,
        n_jobs=4,              # More parallel jobs
        timeout=3600,          # 1 hour timeout
        show_progress_bar=True,
        gc_after_trial=True    # Garbage collect after each trial
    )
    
    return study


study = optuna.create_study(
    direction='minimize',
    storage='sqlite:///hyper_tuning.db',
    study_name='OLS_whole',
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=20,      # More random exploration
        n_ei_candidates=50,       # More candidates for better selection
        gamma=0.25,               # Top 25% trials for modeling
        prior_weight=1.0,         # Weight of prior distribution
        seed=42,                  # seed for reproduce
        multivariate=True,        # Enable multivariate TPE
        group=True,               # Enable categorical grouping
        warn_independent_sampling=True
    ),
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,           # Minimum resource allocation
        max_resource=100,         # Maximum resource allocation  
        reduction_factor=3,       # Resource reduction factor
        bootstrap_count=10        # Number of bootstrap samples
    )
    )
