"""
This file holds different graph generators.
"""
import random

from antz import graph


class GridGraphGenerator(object):
    """
    Implement this one if you generate a grid
    """

    def __init__(self, max_x, max_y, grid_size,
                 node_factory, edge_factory,
                 min_x=0, min_y=0, min_food_hops=10,
                 max_food_hops=50):
        self._max_x = max_x
        self._max_y = max_y
        self._min_x = min_x
        self._min_y = min_y
        self._grid_size = grid_size
        self._node_factory = node_factory
        self._edge_factory = edge_factory
        self._min_food_hops = min_food_hops
        self._max_food_hops = max_food_hops

    def _create_nodes(self):
        nodes = []
        cur_nodes = []

        x = self._min_x
        y = self._min_y

        while True:
            while True:
                node = self._node_factory('waypoint')(x=x, y=y)
                cur_nodes.append(node)
                if x > self._max_x:
                    break
                x += self._grid_size

            nodes.append(cur_nodes)
            cur_nodes = []

            if y > self._max_y:
                break

            x = self._min_x
            y += self._grid_size

        return nodes

    def __call__(self):
        nodes = self._create_nodes()

        g = graph.Graph()

        def create_waypoint(n1, n2):
            wp = self._edge_factory('waypoint')(n1, n2)
            g.add_edge(wp)
            return wp

        yl = len(nodes)
        for i, xlist in enumerate(nodes):
            # make connections from left to right and
            # from top to bottom
            l = len(xlist)
            for j, a in enumerate(xlist):
                ylist = None
                if i + 1 < yl:
                    ylist = nodes[i + 1]
                if j + 1 < l:
                    # create the wayfucker
                    create_waypoint(a, xlist[j + 1])
                    if ylist:
                        # diagonal
                        create_waypoint(a, ylist[j + 1])
                        if j - 1 >= 0:
                            create_waypoint(a, ylist[j - 1])
                if ylist:
                    ylist = nodes[i + 1]
                    b = ylist[j]
                    if a and b:
                        create_waypoint(a, b)

        # add a random nest and food node
        nest_node = random.sample(g.nodes, 1).pop()

        food_node = nest_node

        hops = 0
        if self._min_food_hops == self._max_food_hops:
            max_hops = self._max_food_hops
        else:
            max_hops = random.randrange(self._min_food_hops,
                                        self._max_food_hops)

        # find a food node, respect max hops
        while True:
            next_edges = set([e for e in food_node.edges])

            if not next_edges:
                break

            next_edge = random.sample(next_edges, 1).pop()
            next_node = next_edge.other_node(food_node)

            if not next_node:
                break

            food_node = next_node

            hops += 1

            if hops >= max_hops:
                break

        nest_node.node_type = 'nest'
        food_node.node_type = 'food'

        return nest_node, food_node, g


class RandomGraphGenerator(object):
    """
    Implement this on if you generate a random graph
    """

    def __init__(self, max_x, max_y, num_nodes,
                 node_factory, edge_factory,
                 min_x=0, min_y=0,
                 max_connections=10):
        self._minx = min_x
        self._maxx = max_x
        self._miny = min_y
        self._maxy = max_y
        self._node_factory = node_factory
        self._edge_factory = edge_factory
        self._num_nodes = num_nodes
        self._max_connections = max_connections
        self._fully_connected = max_connections == 0

    def _mk_pos(self):
        x = random.randrange(self._minx, self._maxx)
        y = random.randrange(self._miny, self._maxy)
        pos = (x, y)
        return pos

    def _gen_nodes(self, g):
        node_positions = []
        nest_node, food_node = None, None
        nest_num = random.randrange(0, self._num_nodes)
        food_num = random.randrange(0, self._num_nodes)

        while food_num == nest_num:
            food_num = random.randrange(0, self._num_nodes)

        for i in range(0, self._num_nodes):
            x, y = pos = self._mk_pos()
            while pos in node_positions:
                pos = self._mk_pos()
            node_positions.append(pos)
            node = self._node_factory('waypoint')(x=x, y=y)

            if nest_num == i:
                nest_node = node
                nest_node.nest = True
            elif food_num == i:
                food_node = node
                food_node.food = True

            g.add_node(node)

        return nest_node, food_node

    def __call__(self):
        # create nodes first
        g = graph.Graph()
        nest_node, food_node = self._gen_nodes(g)

        nodes = g.nodes - {nest_node}

        current_node = nest_node
        unconnected_nodes = set(nodes)

        for i in range(0, 50):
            next_edges = set()
            next_nodes = set()

            while unconnected_nodes and current_node != food_node:
                next_node = random.sample(unconnected_nodes, 1).pop()
                if next_node not in next_nodes:
                    next_nodes.add(next_node)
                    edge = self._edge_factory('waypoint')(current_node, next_node)
                    next_edges.add(edge)
                    current_node = next_node
                    unconnected_nodes -= {current_node}

            if next_edges and current_node == food_node:
                if len(next_edges) > self._num_nodes / 5:
                    for e in next_edges:
                        g.add_edge(e)

            unconnected_nodes = set(nodes)

            current_node = nest_node

        return nest_node, food_node, g
