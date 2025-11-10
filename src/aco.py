import logging
from data import Item, TravelTime
from graph import Graph

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
      

  def find_best_path(self):

    # graph.visualize(shortest_path=["A", "B", "C", "D"])

    LOG.info("Finding best path using ACO algorithm...")

# transition rule, evaporation, deposition