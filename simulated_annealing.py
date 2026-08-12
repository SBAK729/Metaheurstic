"""Student file: implement Simulated Annealing here."""

import numpy as np

from experiment import run_experiment

BENCHMARKS_TO_RUN = ["rosenbrock", "sphere", "rastrigin"]
SEEDS = range(100, 120) # Pilot: [0, 1, 2, 3, 4]. Final: range(100, 120).
MAX_EVALUATIONS = 20_000

PARAMETERS = {
    "sigma_fraction": 0.005,     # Gaussian mutation std, as a fraction of domain range
    "n_calibration": 30,        # Random-pair samples used to auto-estimate T0
    "initial_accept_p": 0.8,    # Target acceptance probability at T0
    "t_min_fraction": 1e-4,     # Final temperature, as a fraction of T0
}


def _calibrate_temperature(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    sigma,
    n_calibration,
    initial_accept_p,
):
    """Estimate T0 from typical |delta| between random neighboring points. """

    # Guard against a degenerate calibration budget
    n_calibration = min(n_calibration, max(objective.remaining - 1, 0))

    deltas = []
    x = rng.uniform(lower_bound, upper_bound, size=dimension)
    x_value = objective(x)

    for _ in range(n_calibration):
        noise = rng.normal(loc=0.0, scale=sigma, size=dimension)
        candidate = np.clip(x + noise, lower_bound, upper_bound)
        candidate_value = objective(candidate)
        deltas.append(abs(candidate_value - x_value))

        x, x_value = candidate, candidate_value

    if deltas:
        mean_delta = max(np.mean(deltas), 1e-12) 
        t0 = -mean_delta / np.log(initial_accept_p)
    else:
        # No calibration budget available at all — fall back to a small
        # positive temperature 
        t0 = 1e-6

    return t0, x, x_value


def simulated_annealing(
    objective,
    lower_bound,
    upper_bound,
    dimension,
    rng,
    max_evaluations,
    sigma_fraction=0.05,
    n_calibration=30,
    initial_accept_p=0.8,
    t_min_fraction=1e-3,
    **parameters,
):
    sigma = sigma_fraction * (upper_bound - lower_bound)

    temperature, current, current_value = _calibrate_temperature(
        objective,
        lower_bound,
        upper_bound,
        dimension,
        rng,
        sigma,
        n_calibration,
        initial_accept_p,
    )

    remaining_steps = objective.remaining  # evaluations left after calibration
    t_min = temperature * t_min_fraction
    # Geometric cooling: temperature * cooling_rate**remaining_steps == t_min
    cooling_rate = (t_min / temperature) ** (1.0 / max(remaining_steps, 1))

    while objective.remaining > 0:
        noise = rng.normal(loc=0.0, scale=sigma, size=dimension)
        candidate = np.clip(current + noise, lower_bound, upper_bound)
        candidate_value = objective(candidate)

        delta = candidate_value - current_value
        if delta <= 0 or rng.random() < np.exp(-delta / temperature):
            current, current_value = candidate, candidate_value

        temperature *= cooling_rate


def main():
    for benchmark_name in BENCHMARKS_TO_RUN:
        results = run_experiment(
            simulated_annealing,
            benchmark_name,
            list(SEEDS),
            PARAMETERS,
            max_evaluations=MAX_EVALUATIONS,
        )
        print(f"{benchmark_name}: best objective = {results[0].best_value}")


if __name__ == "__main__":
    main()