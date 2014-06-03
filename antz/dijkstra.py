"""
def Dijkstra(G,start,end=None):
	D = {}	# dictionary of final distances
	P = {}	# dictionary of predecessors
	Q = priorityDictionary()   # est.dist. of non-final vert.
	Q[start] = 0

	for v in Q:
		D[v] = Q[v]
		if v == end: break

		for w in G[v]:
			vwLength = D[v] + G[v][w]
			if w in D:
				if vwLength < D[w]:
					raise ValueError, "Dijkstra: found better path to already-final vertex"
			elif w not in Q or vwLength < Q[w]:
				Q[w] = vwLength
				P[w] = v

	return (D,P)

def shortestPath(G,start,end):
	D,P = Dijkstra(G,start,end)
	Path = []
	while 1:
		Path.append(end)
		if end == start: break
		end = P[end]
	Path.reverse()
	return Path
"""


def shortest_path(graph, source_node, end_node=None):
    """
    Return the shortest path distance between sourceNode and all other nodes
    using Dijkstra's algorithm.  See
    http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm.

    @attention All weights must be nonnegative.

    @type  graph: graph
    @param graph: Graph.

    @type  source_node: node
    @param source_node: Node from which to start the search.

    @rtype  tuple
    @return A tuple containing two dictionaries, each keyed by
        targetNodes.  The first dictionary provides the shortest distance
        from the sourceNode to the targetNode.  The second dictionary
        provides the previous node in the shortest path traversal.
        Inaccessible targetNodes do not appear in either dictionary.
    """
    # Initialization
    dist = {source_node: 0}
    previous = {}
    q = [n for n in graph.nodes]

    # Algorithm loop
    while q:
        # examine_min process performed using O(nodes) pass here.
        # May be improved using another examine_min data structure.
        # See http://www.personal.kent.edu/~rmuhamma/Algorithms/MyAlgorithms/GraphAlgor/dijkstraAlgor.htm
        u = q[0]
        for node in q[1:]:
            if ((not u in dist)
                    or (node in dist and dist[node] < dist[u])):
                u = node

        q.remove(u)

        if u == end_node:
            break

        # Process reachable, remaining nodes from u
        for edge in u.edges:
            v = edge.other_node(u)
            if v in q:
                alt = dist[u] + edge.cost
                if (not v in dist) or (alt < dist[v]):
                    dist[v] = alt
                    previous[v] = u

    return dist, previous