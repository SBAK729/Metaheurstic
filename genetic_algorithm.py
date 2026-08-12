"""Student file: implement a Genetic Algorithm here."""

import numpy as np

from experiment import run_experiment

BENCHMARKS_TO_RUN = ["rosenbrock", "sphere", "rastrigin"]
SEEDS = range(100, 120)  # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

PARAMETERS = {
    "population_size": 20,  # Size of the population each generatio
    "tournament_size": 3,   # Number of competitors sampled per tournament
    "crossover_rate": 0.9,  # Probability that two selected parents actually recombine
    "sigma_fraction": 0.005, # Gaussian mutation
    "mutation_rate": 0.2,   # Probability that any given gene (coordinate) is mutated
}


def _tournament_select(population, values, k, rng):
    """Sample k random individuals, return the best (lowest value) one."""
    competitor_indices = rng.integers(0, len(population), size=k)
    winner_index = competitor_indices[np.argmin(values[competitor_indices])]
    return population[winner_index]


def _blend_crossover(parent1, parent2, dimension, rng):

    alpha = rng.uniform(0.0, 1.0, size=dimension)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


def _mutate(child, sigma, mutation_rate, dimension, lower_bound, upper_bound, rng):

    mutate_mask = rng.random(dimension) < mutation_rate
    noise = rng.normal(loc=0.0, scale=sigma, size=dimension)
    mutated = child + mutate_mask * noise
    return np.clip(mutated, lower_bound, upper_bound)


def genetic_algorithm(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    population_size=40,
    tournament_size=3,
    crossover_rate=0.9,
    sigma_fraction=0.05,
    mutation_rate=0.2,
    **parameters,
):
    sigma = sigma_fraction * (upper_bound - lower_bound)

    # P <- initialize(N)
    init_size = min(population_size, objective.remaining)
    population = rng.uniform(lower_bound, upper_bound, size=(init_size, dimension))
    values = np.array([objective(individual) for individual in population])

    while objective.remaining > 0:

        target_size = min(population_size, objective.remaining)
        children = []

        while len(children) < target_size:
            # p1, p2 <- select(P)
            parent1 = _tournament_select(population, values, tournament_size, rng)
            parent2 = _tournament_select(population, values, tournament_size, rng)

            # c1, c2 <- crossover(p1, p2)
            if rng.random() < crossover_rate:
                child1, child2 = _blend_crossover(parent1, parent2, dimension, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # mutate(c1, c2)
            child1 = _mutate(child1, sigma, mutation_rate, dimension,
                              lower_bound, upper_bound, rng)
            child2 = _mutate(child2, sigma, mutation_rate, dimension,
                              lower_bound, upper_bound, rng)

            # C <- C + [c1, c2]
            children.append(child1)
            if len(children) < target_size:
                children.append(child2)

        children = np.array(children)
        child_values = np.array([objective(child) for child in children])

        population, values = children, child_values



def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            genetic_algorithm,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()