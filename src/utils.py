import logging
import sys
from sklearn.model_selection import ParameterGrid
import pandas as pd
from aco import ACO


LOG = logging.getLogger(__name__)


def setup_logging() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
  )

def hyperparam_tuning(items, travel_times):
  param_grid = {
    "pheromones": [0.5, 1.0, 2],
    "heuristic": [2.0, 4.0],
    "rho": [0.05, 0.1],
    "ants": [30, 50],
    "iters": [80, 120],
    "Q": [5.0, 10.0, 15.0],
  }
  seeds = [0, 1, 2]
  best_result = None
  all_trials: list[dict] = []

  for params in ParameterGrid(param_grid):
    costs = []
    best_run = None
    for seed in seeds:
      aco = ACO(items=items, travel_times=travel_times)
      aco.construct_graph()
      path, cost = aco.find_best_path(seed=seed, **params)
      costs.append(cost)
      if best_run is None or cost < best_run["cost"]:
        best_run = {"path": path, "cost": cost, "seed": seed}

    avg_cost = sum(costs) / len(costs)
    candidate = {
      "params": params,
      "avg_cost": avg_cost,
      "path": best_run["path"] if best_run else None,
      "best_cost": best_run["cost"] if best_run else float("inf"),
      "seed": best_run["seed"] if best_run else None,
    }
    all_trials.append({**params, "avg_cost": avg_cost, "best_cost": candidate["best_cost"]})
    if best_result is None or avg_cost < best_result["avg_cost"]:
      best_result = candidate

  if all_trials:
    df = pd.DataFrame(all_trials).sort_values("avg_cost")
    LOG.info("Hyperparameter grid (sorted by avg_cost):\n%s", df.to_string(index=False))

  return best_result
