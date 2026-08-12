"""Student file: implement an Evolution Strategy here."""

import numpy as np

from experiment import run_experiment

BENCHMARKS_TO_RUN = ["rosenbrock", "sphere", "rastrigin"]
SEEDS = range(100, 120)  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

PARAMETERS = {
    "mu": 8,                    # Number of parents kept each generation
    "lam": 40,                  # Number of children produced each generation
    "sigma_fraction": 0.01,     # Gaussian mutation
    "plus_selection": False,    # True -> (mu+lambda),  False -> (mu,lambda)
}


def evolution_strategy(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    mu=5,
    lam=20,
    sigma_fraction=0.1,
    plus_selection=False,
    **parameters,
):
    sigma = sigma_fraction * (upper_bound - lower_bound)

    init_size = min(lam, objective.remaining)
    population = rng.uniform(lower_bound, upper_bound, size=(init_size, dimension))
    values = np.array([objective(individual) for individual in population])

    while objective.remaining > 0:

        # parents <- best mu from P  (truncation selection)
        k = min(mu, len(population))
        best_indices = np.argsort(values)[:k]
        parents = population[best_indices]
        parent_values = values[best_indices]

        # children <- mutate(parents, lambda)
        batch_size = min(lam, objective.remaining)
        parent_choices = rng.integers(0, k, size=batch_size)
        noise = rng.normal(loc=0.0, scale=sigma, size=(batch_size, dimension))
        children = np.clip(parents[parent_choices] + noise, lower_bound, upper_bound)
        child_values = np.array([objective(child) for child in children])

        if plus_selection:
            population = np.vstack([parents, children])
            values = np.concatenate([parent_values, child_values])
        else:
            population = children
            values = child_values


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            evolution_strategy,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()