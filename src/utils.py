import logging
import sys

def setup_logging() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
  )

def _neighbors(self, u:str) -> list[tuple[str, int]]:
  """Return list of valid outgoing edges (v, duration) from current node u"""
  nxg = self.graph._nxGraph
  if u not in nxg:
    return []
  out = []
  for v in nxg.successors(u):
    data = nxg[u][v]
    dur = data.get("duration", -1)
    if isinstance(dur, (int, float)) and dur >= 0:
      out.append((v, int(dur)))
  return out

def _heuristic(self, duration: int) -> float:
  # prefer shorter path, defined by epsilon
  return 1.0/(duration + 1e-6)
  
def _choose_next(self, u: str, pheromones: float, heuristic: float, rng: random.Random) -> Optional[str]:
  """Stochastic next-node choice based on pheromonse and heuristics"""

  nxg = self.graph_nxgraph
  neighbors = self._neighbors(u)
  if not neighbors:
    return None
  
  weights = []
  candidates = []
  for v, dur in neighbors:
    tau = float(nxg[u][v].get("pheromones", 1.0))
    eta = self._heuristic(dur)
    w = (tau ** pheromones) * (eta ** heuristic)
    if w > 0.0:
      candidates.append(v)
      weights.append(w)
  if not candidates:
    return None
  

  """roulette wheel target picking, influenced by weight"""
  total = sum(weights)
  r = rng.random()*total #r is a random number from 0 to total
  acc = 0.0
  for v, w in zip(candidates, weights):
    acc += w
    if r <= acc:
      return v
  return candidates[-1]

def shortest_path(self) -> Tuple[List[str], [int]]:
"""Dijkstra baseline on edge attribute 'duration"""
import networkx as nx


