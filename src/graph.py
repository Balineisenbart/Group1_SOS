import networkx as nx

from typing import List
import matplotlib.pyplot as plt

class Graph:
  _nxGraph: nx.DiGraph

  def __init__(self) -> None:
    self._nxGraph = nx.DiGraph()

  def add_node(self, node_id: str, **attrs) -> None:
    self._nxGraph.add_node(node_id, **attrs)
  
  def add_edge(self, from_id: str, to_id: str, **attrs) -> None:
    self._nxGraph.add_edge(from_id, to_id, **attrs)
        
  def visualize(self, shortest_path: List[str]) -> None:
    for edge in self._nxGraph.edges:
        source, destination = edge[0], edge[1]
        self._nxGraph[source][destination]["pheromones"] = round(
            self._nxGraph[source][destination]["pheromones"]
        )

    default_size = plt.rcParams.get("figure.figsize", (6.4, 4.8))
    width, height = default_size
    plt.figure(figsize=(width * 6, height))

    x_gap = 20
    y_gap = 20
    plt.xlim(-x_gap, x_gap * 25)  # zoom out horizontally
    plt.ylim(-y_gap, y_gap * 3)

    pos = {}

    for node in self._nxGraph.nodes(data=True):
        district = node[1]["district"]
        index_in_district = sum(
            1
            for n in self._nxGraph.nodes(data=True)
            if n[1]["district"] == district and n[0] < node[0]
        )

        pos[node[0]] = (district * x_gap, index_in_district * y_gap)

    pos["START"] = (0, 1 * y_gap)
    pos["END"] = (24 * x_gap, 1 * y_gap)

    
    nx.draw(self._nxGraph, pos, width=2)

    nx.draw_networkx_nodes(self._nxGraph, pos, node_size=20, node_shape="D")
    nx.draw_networkx_edges(
        self._nxGraph,
        pos,
        edgelist=list(zip(shortest_path, shortest_path[1:])),
        edge_color="r",
        width=2,
    )

    nx.draw_networkx_labels(self._nxGraph, pos, font_size=4)
    # edge_labels = nx.get_edge_attributes(self._nxGraph, "pheromones")
    edge_labels = nx.get_edge_attributes(self._nxGraph, "duration")
    nx.draw_networkx_edge_labels(self._nxGraph, pos, edge_labels, label_pos=0.6, font_size=6)

    ax = plt.gca()
    ax.margins(0.00, 0.05)
    plt.axis("off")
    plt.tight_layout()
    plt.show()