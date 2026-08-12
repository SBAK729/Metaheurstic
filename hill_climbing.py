"""Student file: implement Random-Restart Hill Climbing here."""

import numpy as np

from experiment import run_experiment

# For a quick test, choose one benchmark and one seed, then run this file.
BENCHMARKS_TO_RUN = ["rosenbrock", "sphere", "rastrigin"]
SEEDS = range(100, 120)
MAX_EVALUATIONS = 20_000


PARAMETERS = {
    "n_restarts": 50, #how many independent hill climbs
    "sigma_fraction": 0.01, #Gaussian mutation
}


def _single_climb(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    budget,
    sigma,
):
    """Run one greedy hill climb, consuming exactly `budget` evaluations."""
    parent = rng.uniform(lower_bound, upper_bound, size=dimension)
    parent_value = objective(parent)
    used = 1

    while used < budget:
        noise = rng.normal(loc=0.0, scale=sigma, size=dimension)
        child = np.clip(parent + noise, lower_bound, upper_bound)

        child_value = objective(child)
        used += 1

        # Greedy acceptance: only move to equal-or-better neighbors.
        if child_value <= parent_value:
            parent = child
            parent_value = child_value


def hill_climbing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    n_restarts=40,
    sigma_fraction=0.05,
    **parameters,
):
    sigma = sigma_fraction * (upper_bound - lower_bound)
    restart_budget = max(1, max_evaluations // n_restarts)

    while objective.remaining > 0:
        budget_this_restart = min(restart_budget, objective.remaining)
        _single_climb(
            objective,
            lower_bound,
            upper_bound,
            dimension,
            rng,
            budget_this_restart,
            sigma,
        )


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            hill_climbing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()