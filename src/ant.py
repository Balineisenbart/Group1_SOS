from graph import Graph
import random

class Ant:

  graph: Graph

  source: str
  destination: str

  path: list[str]
  cost: int = 0

  def __init__(self, graph: Graph, source: str, destination: str) -> None:
    self.graph = graph
    self.source = source
    self.destination = destination
    self.path = [source]

  def deploy(self) -> None:
    while not self.has_reached_destination():
      self.step()

  def step(self) -> None:
    next_node = self._choose_next_node()

    self.cost += self.graph.get_cost(self._get_current_node(), next_node)
    self.path.append(next_node)

  def _choose_next_node(self) -> str:
    neighbors = self.graph.neighbors(self._get_current_node())
    print(neighbors)

    target = random.choice(neighbors) # TODO: implement ant algo

    return target

  def _get_current_node(self) -> str:
    return self.path[-1]

  def has_reached_destination(self) -> bool:
    return self._get_current_node() == self.destination

