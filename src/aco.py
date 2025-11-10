import logging
from data import Item, TravelTime
from graph import Graph
import random
from typing import List, Tuple, Optional

LOG = logging.getLogger(__name__)



class ACO:
  items: list[Item]
  travel_times: list[TravelTime]

  graph = Graph()

  def __init__(self, items, travel_times) -> None:
    LOG.info("Initializing ACO...")
    self.items = items
    self.travel_times = travel_times

  def construct_graph(self):
    LOG.info("Constructing graph for ACO...")

    self.graph.add_node("START", name="START", district=0)
    self.graph.add_node("END", name="END", district=24)

    for item in self.items:
      self.graph.add_node(item.id, name=item.name, district=item.district)

      if item.district == 1:
        self.graph.add_edge( "START", item.id, pheromones=1.0, duration=0)

      if item.district == 23:
        self.graph.add_edge(item.id, "END", pheromones=1.0,duration=0)

      for target in self.items:
        if target.district == item.district + 1:

          # find with from_id and to_id in travel_times
          travel_time = next(
              (tt for tt in self.travel_times if tt.from_id == item.id and tt.to_id == target.id),
              None
          )

          print(item.id, "->", target.id, ":", travel_time)

          self.graph.add_edge(
              item.id,
              target.id,
              pheromones=1.0,
              duration=travel_time.duration if travel_time else -1,
          )
      

def find_best_path(self, 
                    pheromones: float = 1.0,
                    heuristic: float = 3.0, 
                    rho: float = 0.1,
                    ants: int = 30,
                    iters: int = 100,
                    Q: float = 10.0,
                    seed: int = 0,
                    use_elitist: bool = True
                    )-> Tuple[List[str], int]:

  # graph.visualize(shortest_path=["A", "B", "C", "D"])
  """Returns (best_path, best_cost). """

  LOG.info("Finding best path using ACO algorithm...")

  rng = random.Random(seed)
  nxg = self.graph._nxGraph

  try: 
    best_path, best_cost = self.shortest_path()
  except Exception as e:
    raise RuntimeError("Graph is not connected from START to END") from e

  for it in range(iters):
    iter_best_path: Optional[List[str]] = None
    iter_best_cost = float("inf")

#here the ant paths are constructed
    for _ in range(ants):
      u = "START"
      path = ["START"]
      steps = 0
      ok = True
      while u != "END":
        v = self._choose_next(u, pheromones, heuristic, rng)
        if v is None:
          ok = False
          break
        path.append(v)
        u = v
        steps += 1
        if steps > 24:
          ok = False
          break
      if not ok or path[-1] != "END":
        continue

      cost = self._path_cost(path)
      if cost < iter_best_cost:
        iter_best_cost = cost
        iter_best_path = path

#here evaporation is introduced - establishing optimization
    for a, b in nxg.edges:
      tau = float(nxg[a][b].get("pheromones", 1.0))
      nxg[a][b]["pheromones"] = max(1e-12, (1.0 - rho) * tau)

#refreshing of pheromones - node selection reinforcement
    if iter_best_path is not None and iter_best_cost > 0:
      deposit = Q / float(iter_best_cost)
      for a, b in zip(iter_best_path, iter_best_path[1:]):
        nxg[a][b]["pheromones"] = float(nxg[a][b]["pheromones"]) + deposit

      if iter_best_cost < best_cost:
        best_cost = int(iter_best_cost)
        best_path = list(iter_best_path)
    
    LOG.debug(
      "iter=%d iter_best=%s global_best=%s",
      it,
      iter_best_cost,
      best_cost,
    )

  return best_path, best_cost