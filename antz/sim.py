"""
"""

import math
import bisect
import random
import collections
import operator
from uuid import uuid4

from antz import graph as antz_graph
from antz.util import *


DEBUG = True


class AlgorithmContext(object):
    def __init__(self, graph, state):
        self.graph = graph
        self.state = state


class Algorithm(object):
    """
    We need something like a behavior because we want to solve
    different problems with the ants.

    Such as:

        - Find the shortest path
        - Solve the TSP
        - Do war

    The behavior should be an attribute of the ant, like a part
    of the brain. We don't want subclasses here because those would
    be too static.
    """

    TYPE = None  # should be set by the implementation

    def begin_round(self, ctx):
        """
        Begin a new round
        """

    def end_round(self, ctx):
        """
        Begin a new round
        """

    def init_ant(self, ctx, ant):
        """
        Initialize the ant
        """

    def choose_edge(self, ctx, ant, edges):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

    def visit_edge(self, ctx, ant, edge):
        """
        Called when the ant moves over the edge to another node.
        """

    def leave_edge(self, ctx, ant, edge):
        """
        Called when the ant leaves an edge.
        """

    def visit_node(self, ctx, ant, node):
        """
        Called when the ant visits a node.
        """

    def leave_node(self, ctx, ant, node):
        """
        Called when the ant leaves a node.
        """

    def end_turn(self, ctx, ant):
        """
        Called after one turn
        """

    def create_state(self):
        """
        Returns a state object which has to be passed on every
        method call.
        """

    def create_ant_state(self):
        """
        Returns a state object which should be attaced to an ant
        """

    def __repr__(self):
        """
        Default repr implementation for convenience 
        """
        return ('%s(TYPE=%s, <%s>)'
                % (self.__class__.__name__,
                   self.TYPE, id(self)))


def relative_cost(g, edge):
    if edge.cost == 0:
        raise ValueError()
    return edge.cost / g.max_cost


def get_weighted_choice(probabilities):
    if probabilities:
        cdf = [probabilities[0][1]]
        for i in range(1, len(probabilities)):
            cdf.append(cdf[i - 1] + probabilities[i][1])
        r = random.random()
        ind = bisect.bisect(cdf, r)
        item = probabilities[ind]
        return item[0]


class Attribute(object):
    def __init__(self, name, prop, converter, default=None):
        self.name = name
        self.prop = prop
        self.converter = converter
        self.default = default


class Reset(Exception):
    pass


class ShortestPathAlgorithm(Algorithm):
    """
    Makes the and behave in a way which leads to a shortes path solution
    through n nodes.
    """

    ALPHA = 2
    BETA = 2
    GAMMA = 1
    PHERO_DECREASE = 0.01
    EXISTING_DECREASE_POW = 0
    PHERO_UPDATE_INSTANT = False

    # type and attributes are used by ui to show
    # additional configuration options
    NAME = 'Shortest Path'
    TYPE = 'shortest_path'
    ATTRIBUTES = [
        Attribute('Alpha', 'alpha', float, default=ALPHA),
        Attribute('Beta', 'beta', float, default=BETA),
        Attribute('Gamma', 'gamma', float, default=GAMMA),
        Attribute('Pheromone Decrease', 'phero_decrease', float, default=PHERO_DECREASE),
        Attribute('Existing Decrease Exp', 'existing_decrease_pow', float, default=EXISTING_DECREASE_POW),
    ]

    class AlgorithmState(object):
        def __init__(self):
            self._solutions = []
            self._pheromone_edges = set()
            self._solution_counts = collections.defaultdict(int)
            self._best_solution = None
            self._edge_probs = {}
            self._max_phero = 0
            self._local_best_solution = None
            self._rounds = 0

        @property
        def solutions(self):
            return self._solutions

        @property
        def rounds(self):
            return self._rounds

        @property
        def best_solution(self):
            return self._best_solution

        @property
        def local_best_solution(self):
            return self._local_best_solution

        @property
        def updated_edges(self):
            return self._pheromone_edges

    class AntState(object):
        def __init__(self):
            self.edges = []
            self.nodes = set()
            self.turns = 0
            self.solution = None
            self.solution_length = 0

        def add_node(self, node):
            self.nodes.add(node)

        def add_edge(self, edge):
            self.edges.append(edge)
            self.solution_length += edge.cost

    def create_state(self):
        """
        Returns a state object which has to be passed on every
        method call.
        """
        return self.__class__.AlgorithmState()

    def create_ant_state(self):
        """
        Returns a state object which should be attaced to an ant
        """
        return self.__class__.AntState()

    def __init__(self):
        self.alpha = self.ALPHA
        self.beta = self.BETA
        self.gamma = self.GAMMA
        self.phero_decrease = self.PHERO_DECREASE
        self.existing_decrease_pow = self.EXISTING_DECREASE_POW
        self.phero_update_instant = self.PHERO_UPDATE_INSTANT

    def choose_edge(self, ctx, ant, node):
        """
        Function should choose an edge among the passed 
        edges and return it. 
        """

        state = ant.state

        if state.solution:
            return state.edges.pop()
        else:
            edges = self._get_edges_to_consider(node, exclude_nodes=state.nodes,
                                                exclude_edges=state.edges)

            if not edges:
                return

            probabilities = ctx.state._edge_probs.get(node)
            if not probabilities:
                pkind = self._get_default_pheromone_kind(ant)

                pheromone_levels = [(edge, edge.pheromone_level(pkind))
                                    for edge in edges]
                pheromone_levels_only = [level for _, level in pheromone_levels]

                min_level = 0
                max_level = max(pheromone_levels_only)

                min_cost = 0
                max_cost = min([edge.cost for edge in edges])

                # noinspection PyShadowingNames
                def calculate_probability(cost, level, min_level, max_level,
                                          min_cost, max_cost):
                    level = (level - min_level) / (max_level - min_level)
                    cost = 1 / ((cost - min_cost) / (max_cost - min_cost))
                    level **= self.alpha
                    cost **= self.beta
                    level *= cost
                    return level

                probabilities = [(edge, calculate_probability(edge.cost, level, min_level,
                                                              max_level, min_cost, max_cost))
                                 for edge, level in pheromone_levels]

                probablity_sum = sum([prob for _, prob in probabilities])
                probablity_sum = probablity_sum or 1

                probabilities = [(edge, prob / probablity_sum)
                                 for edge, prob in probabilities]

                ctx.state._edge_probs[node] = probabilities

            return get_weighted_choice(probabilities)

    def _evaporate(self, ctx):
        # to_remove = set()
        for edge in ctx.state._pheromone_edges:
            store = edge.pheromone_store
            kinds = store.kinds
            for k in kinds:
                level = store.get(k)
                store.set(Pheromone(k, (1.0 - self.phero_decrease) * level))

    def visit_edge(self, ctx, ant, edge):
        """
        Just drop some pheromone on the edge
        """

        state = ant.state

        if not state.solution:
            state.add_edge(edge)
        else:
            current_phero = edge.pheromone_level(
                ant.pheromone_kind('default'))

            min_phero = 0
            max_phero = ctx.state._max_phero

            phero_inc = (1 / state.solution[1]) ** self.gamma

            if current_phero and max_phero:
                existing_multiplicator = ((1 - (current_phero - min_phero) / (max_phero - min_phero))
                                          ** self.existing_decrease_pow)
                phero_inc *= existing_multiplicator

            edge.increase_pheromone(
                ant.create_pheromone(
                    'default', phero_inc))

            current_phero = edge.pheromone_level(ant.pheromone_kind('default'))
            if current_phero > ctx.state._max_phero:
                ctx.state._max_phero = current_phero

            ctx.state._pheromone_edges.add(edge)

    def _add_solution(self, ctx, ant):
        # increase the solution count to check for most accessed
        # solutions at the end of the round
        if ant.state.solution:
            ctx.state._solution_counts[ant.state.solution] += 1

    def visit_node(self, ctx, ant, node):
        """
        Called when a node is visited. We mainly check wether we
        hit a food or a nest node and react accordingly.
        """

        if not self._can_pass(node):
            # if the ant hits an obstacle, kill it
            raise Reset()
        else:
            ant.state.add_node(node)

            solution = (tuple(ant.state.edges), ant.state.solution_length)

            if node_is_food(node):
                # the ant found a solution
                ant.state.solution = solution
                self._add_solution(ctx, ant)
            elif node_is_nest(node):
                # when it's a nest and we are on our way
                # back -> reset the ant
                if ant.state.solution:
                    raise Reset()

            if ant.state.solution:
                ctx.state._solution_counts[ant.state.solution] += 1

    def end_turn(self, ctx, ant):
        """
        Check whether the ant is hopelessly running around and
        not finding a solution. Let it die with a certain probability.
        """

        state = ant.state
        state.turns += 1

        rand = random.random()
        if not state.solution:
            if ctx.state._best_solution:
                if rand < 0.2 and state.turns > random.randrange(200, 400):
                    raise Reset()

    def _is_path_accessible(self, path):
        """
        Checks whether the passed path only consists of accessible nodes.
        """
        if not path:
            return False

        for edge in path:
            n1 = edge.node_from
            n2 = edge.node_to
            if not self._can_pass(n1) or not self._can_pass(n2):
                return False
        return True

    def begin_round(self, ctx):
        """
        Initialize some values which we create by round. This is the
        probability cache and the counts for active solutions.
        """
        ctx.state._edge_probs = {}
        ctx.state._solution_counts = collections.defaultdict(int)

    def end_round(self, ctx):
        """
        Determine the active, the best and the mostly used solutions.

        Evaporate the pheromone trails.
        """
        ctx.state._rounds += 1

        if ctx.state._solution_counts:
            ctx.state._solutions = sorted(ctx.state._solution_counts.keys(),
                                          key=lambda item: item[1])
            best_solution = ctx.state._solutions[0]
            if not ctx.state._best_solution:
                ctx.state._best_solution = best_solution
            else:
                if best_solution[1] < ctx.state._best_solution[1]:
                    ctx.state._best_solution = best_solution

            ctx.state._local_best_solution = best_solution

        if ctx.state._local_best_solution:
            path, _ = ctx.state._local_best_solution
            if not self._is_path_accessible(path):
                ctx.state._local_best_solution = None

        if ctx.state._best_solution:
            path, _ = ctx.state._best_solution
            if not self._is_path_accessible(path):
                ctx.state._best_solution = None

        # evaporate pheromones
        self._evaporate(ctx)

    def _can_pass(self, node):
        """
        Returns whether this noce can be passed by an ant.

        ATM checks only whether the node is an obstacle or not.
        """
        return not node_is_obstacle(node)

    def _get_edges_to_consider(self, node, exclude_nodes=None, exclude_edges=None):
        """
        Returns edges which are not yet accessed by the ant (passed by
        `exclude_edges` and which are accessible (not obstacles).
        """

        edges = node.edges
        exclude_nodes = exclude_nodes or set()
        exclude_edges = exclude_edges or set()

        edges = [e for e in edges
                 if e.other_node(node) not in exclude_nodes
                 and e not in exclude_edges]

        edges = [e for e in edges
                 if self._can_pass(e.other_node(node))]

        return edges

    def _get_default_pheromone_kind(self, ant):
        """
        Returns the default kind of pheromone of the ants colony
        """
        return ant.pheromones.pheromone_kind('default')


class Ant(object):
    """
    Represents the actual agent which is mapped to a ui entity.

    Ant ant is driven by the algorithm passed to it.
    """

    def __init__(self, initial_node, pheromones, state):
        """
        The ant is in fact a small wrapper around :cls:`.AntBehavior`.

        The behavior does the decisions and interactions with the environment.
        """

        self._pheromones = pheromones
        self._current_node = initial_node
        self._current_edge = None
        self._path_length = 0

        # store your state here
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def pheromones(self):
        return self._pheromones

    def reset(self, initial_node, state=None):
        """
        Resets the ant to initial state so it can be reused.
        """

        self._state = state
        self._current_node = initial_node
        self._current_edge = None

    def pheromone_kind(self, kind):
        return self.pheromones.pheromone_kind(kind)

    def create_pheromone(self, kind, amount):
        """
        Create a pheromone of the passed kind.

        Pass 'default' for the default kind.
        """
        kind = self.pheromones.pheromone_kind(kind)
        return Pheromone(kind, amount)

    @property
    def current_edge(self):
        """
        Returns the current edge the ant stays on
        """
        return self._current_edge

    @property
    def current_node(self):
        """
        Returns the current node the ant stays on
        """
        return self._current_node

    def move(self, algorithm, ctx):
        """
        Move to the next node. This is entirely driven by algorithm.
        """

        current_node = self.current_node

        edge = algorithm.choose_edge(ctx, self, current_node)

        if edge:
            self._current_edge = edge

            # returns the node which is not the current nod
            next_node = edge.other_node(current_node)

            # if the edge is unidirectional
            if next_node:
                if self._current_edge:
                    algorithm.leave_edge(
                        ctx, self, self._current_edge)

                self._current_node = next_node

                algorithm.leave_node(ctx, self, current_node)
                algorithm.visit_edge(ctx, self, edge)
                algorithm.visit_node(ctx, self, next_node)
            else:
                # reset the ant
                raise Reset()

            algorithm.end_turn(ctx, self)
        else:
            raise Reset()


def node_is_food(node):
    """ Check whether a node is food """
    return node.food


def node_is_nest(node):
    """ Check whether a node is a nest """
    return node.nest


def node_is_waypoint(node):
    """ Check whether a node is a waypoint """
    return node.waypoint


def node_is_obstacle(node):
    """ Check whether a node is an obstacle """
    return node.obstacle


class Waypoint(antz_graph.Node):
    def __init__(self, x=None, y=None, name=None, node_type='waypoint'):
        antz_graph.Node.__init__(self, name=name)
        self._x = x
        self._y = y
        self._node_type = node_type
        self._obstacle = False

    @property
    def node_type(self):
        return self._node_type

    @node_type.setter
    def node_type(self, node_type):
        self._node_type = node_type

    @property
    def food(self):
        return self._node_type == 'food'

    @food.setter
    def food(self, food):
        if food:
            self._node_type = 'food'

    @property
    def nest(self):
        return self._node_type == 'nest'

    @nest.setter
    def nest(self, nest):
        if nest:
            self._node_type = 'nest'

    @property
    def waypoint(self):
        return self._node_type == 'waypoint'

    @waypoint.setter
    def waypoint(self, waypoint):
        if waypoint:
            self._node_type = 'waypoint'

    @property
    def obstacle(self):
        return self._obstacle

    @obstacle.setter
    def obstacle(self, obstacle):
        self._obstacle = obstacle

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def _calculate_distance_to(self, other):
        if self.x is None or self.y is None:
            return None

        x1, y1, x2, y2 = self.x, self.y, other.x, other.y
        x = abs(abs(x1) - abs(x2))
        y = abs(abs(y1) - abs(y2))
        distance = math.sqrt(x ** 2 + y ** 2)

        return distance

    def __repr__(self):
        return ('%s(TYPE=%s, name=%s)'
                % (self.__class__.__name__,
                   self.node_type, self._name))


class WaypointEdge(antz_graph.Edge):
    """
    Our edge implementation with pheromones
    """

    def __init__(self, node_from, node_to, pheromone_store=None, **kwargs):
        antz_graph.Edge.__init__(self, node_from, node_to, **kwargs)
        self._pheromone_store = pheromone_store or PheromoneStore()

    @property
    def pheromone_store(self):
        return self._pheromone_store

    def pheromone_level(self, kind):
        return self._pheromone_store.get(kind) or 10 ** -10

    def increase_pheromone(self, pheromone):
        self._pheromone_store.increase(pheromone)

    def decrease_pheromone(self, pheromone):
        self._pheromone_store.decreases(pheromone)

    def set_pheromone_level(self, pheromone):
        self._pheromone_store.set(pheromone)

    def __repr__(self):
        return ('%s(node_from=%s, node_to=%s)'
                % (self.__class__.__name__,
                   self.node_from, self.node_to))


class PheromoneStore(object):
    def __init__(self):
        self._level = {}

    def clear(self):
        self._level = {}

    @property
    def kinds(self):
        return self._level.keys()

    def increase(self, pheromone):
        k = pheromone.kind
        self._set(k, self._get(k) + pheromone.amount)

    def _decrease(self, kind, amount):
        self._set(kind, self._get(kind) - amount)

    def _get(self, kind):
        level = self._level.get(kind, 0.0)
        #print(level)
        return level

    def _set(self, kind, amount):
        self._level[kind] = amount

    def get(self, kind):
        return self._get(kind)

    def set(self, pheromone):
        self._set(pheromone.kind, pheromone.amount)

    def decrease(self, pheromone):
        self._decrease(pheromone.kind, pheromone.amount)


class Pheromone(object):
    def __init__(self, kind, amount):
        self._kind = kind
        self._amount = amount

    @property
    def kind(self):
        return self._kind

    @property
    def amount(self):
        return self._amount


class PheromoneKind(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class Pheromones(object):
    def __init__(self, name=None):
        name = name or str(uuid4())
        # make a set out of the ants
        self._pheromone_kinds = {
            'default': PheromoneKind(name='%s:%s' % (name, 'default'))
        }

    def pheromone_kind(self, kind):
        return self._pheromone_kinds[kind]


def _reset_ant(ant, algorithm, nest_node):
    ant.reset(nest_node, algorithm.create_ant_state())
    return ant


class Runner(object):
    """
    Returned by AntColony.create_runner
    """

    def __init__(self, ant_factory, nest_node, algorithm, graph,
                 num_ants=1000):
        self._nest_node = nest_node
        self._state = algorithm.create_state()
        self._algorithm = algorithm
        self._ctx = AlgorithmContext(graph, self._state)
        self._ant_factory = ant_factory
        self._num_ants = num_ants
        self._ants = set([self._ant_factory()
                          for _ in range(0, self._num_ants)])

    def _reset_ant(self, ant):
        """
        Resets an ant in one run. To reset the whole run,
        you just have to create a new runner
        """
        _reset_ant(ant, self._algorithm, self._nest_node)

    @property
    def ants(self):
        return self._ants

    @property
    def solutions(self):
        return self._state.solutions

    @property
    def rounds(self):
        return self._state.rounds

    @property
    def best_solution(self):
        return self._state.best_solution

    @property
    def local_best_solution(self):
        return self._state.local_best_solution

    @property
    def updated_edges(self):
        return self._state.updated_edges

    def create_ant(self):
        ant = self._ant_factory()
        self._ants.add(ant)
        return ant

    def remove_ant(self, ant):
        self._ants.remove(ant)

    def next_step(self):
        """
        Move all ants
        """
        self._algorithm.begin_round(self._ctx)
        for ant in self._ants:
            try:
                ant.move(self._algorithm,
                         self._ctx)
            except Reset:
                self._reset_ant(ant)
        self._algorithm.end_round(self._ctx)


class AntColony(object):
    def __init__(self, num_ants=1000):
        self._num_ants = num_ants
        self._pheromones = Pheromones()

    @property
    def pheromones(self):
        return self._pheromones

    def pheromone_kind(self, name):
        return self._pheromones.pheromone_kind(name)

    def create_runner(self, algorithm, graph, nest_node, num_ants=1000):
        for edge in graph.edges:
            edge.pheromone_store.clear()

        def ant_factory():
            return Ant(nest_node, self._pheromones, algorithm.create_ant_state())

        return Runner(ant_factory, nest_node, algorithm, graph,
                      num_ants=num_ants)

