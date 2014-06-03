import sets
import collections


class Node(object):
    def __init__(self, name=None):
        self._graph = None
        self._name = name
        self._edges = None

    @property
    def name(self):
        return self._name

    @property
    def edges(self):
        """
        This function returns the edges for this node.
        Will be patched by the graph.
        """
        if self._graph and not self._edges:
            self._edges = self._graph.get_edges(self)
        return self._edges

    def distance_to(self, other):
        return self._calculate_distance_to(other)

    def _calculate_distance_to(self, other):
        return 0

    def __sub__(self, other):
        if not isinstance(other, Node):
            raise TypeError()
        return self.distance_to(other)


class Edge(object):
    """
    Represents the connection between two nodes
    """

    def __init__(self, node_from, node_to, bidirectional=True, cost=()):
        if not node_from or not node_to:
            raise ValueError('Both nodes must be set')
        self._node_from = node_from
        self._node_to = node_to
        self._nodes = {
            node_from: node_to,
            node_to: node_from
        }
        self._node_set = {node_from, node_to}
        self._bidirectional = bidirectional
        if cost is ():
            cost = node_to - node_from
        cost = cost or 0
        self._cost = cost

    @property
    def node_from(self):
        return self._node_from

    @property
    def node_to(self):
        return self._node_to

    @node_from.setter
    def node_from(self, node):
        self._node_from = node

    @node_to.setter
    def node_to(self, node):
        self._node_to = node

    @property
    def bidirectional(self):
        return self._bidirectional

    @property
    def cost(self):
        return self._cost

    def other_node(self, current_node):
        """
        Returns the other node excluding current one. Does
        take bidirectional into account. If you may not move
        in the direction of other node, None will be returned.
        """
        return self._nodes[current_node]

    @property
    def nodes(self):
        return self._node_set


class Graph(object):
    """
    Graph which holds the possible movements of ants
    """

    def __init__(self):
        self._nodes = set()
        self._edges = set()
        self._node_edges = collections.defaultdict(dict)
        self._min_cost = float('inf')
        self._max_cost = float('-inf')

    def get_edges(self, node):
        conns = self._node_edges[node]
        return conns.values()         

    def add_node(self, node):
        """
        Add a new node to the graph
        """
        if node not in self._nodes:
            node._graph = self
            self._nodes.add(node)

    def add_edge(self, edge):
        node_edges = self._node_edges

        self._edges.add(edge)

        self.add_node(edge.node_from)    
        self.add_node(edge.node_to)

        node_edges[edge.node_from][edge.node_to] = edge
        if edge.bidirectional:
            node_edges[edge.node_to][edge.node_from] = edge

        if edge.cost > self._max_cost:
            self._max_cost = edge.cost
        if edge.cost < self._min_cost:
            self._min_cost = edge.cost

    @property
    def min_cost(self):
        return self._min_cost

    @property
    def max_cost(self):
        return self._max_cost

    def remove_edge(self, params):
        """
        Either pass one param: an edge or pass two params
        which will delete the edge between those nodes.
        """
        if len(params) == 1:
            edge = params[0]
        elif len(params) == 2:
            edge = self._node_edges[params[0]][params[1]]
        else:
            raise TypeError('Provide either one or two parameters.')
        self._edges.remove(edge)
        n1, n2 = edge.node_from, edge.node_to
        del self._node_edges[n1][n2]
        if edge.bidirectional:
            del self._node_edges[n2][n1]

    def remove_node(self, node):
        # clean edges
        # remove from nodes first
        self._nodes.remove(node)
        edges = self._node_edges[node]
        for node2 in edges.items():
            # clean the backreferences
            del self._node_edges[node2][node]
        del self._node_edges[node]
    
    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        """
        Returns an immutable set. Maybe we can add mutable
        properties if we really need it. But I don't think so.
        """
        return self._edges


def generate_connected_graph(num_nodes):
    """
    """