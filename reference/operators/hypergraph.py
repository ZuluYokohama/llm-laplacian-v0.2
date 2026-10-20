"""Normalized hypergraph Laplacian.

L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}.
D_e is the vertex count of each hyperedge. D_v sums incident edge weights.
"""


def hypergraph_laplacian(incidence, edge_weights):
    vertices = len(incidence)
    edges = len(incidence[0])
    edge_degree = []
    for edge in range(edges):
        edge_degree.append(sum(incidence[vertex][edge] for vertex in range(vertices)))
    vertex_degree = []
    for vertex in range(vertices):
        total = 0.0
        for edge in range(edges):
            total += incidence[vertex][edge] * edge_weights[edge]
        vertex_degree.append(total)
    # Middle = W D_e^{-1} H^T D_v^{-1/2}, shape edges by vertices.
    middle = [[0.0 for _ in range(vertices)] for _ in range(edges)]
    for edge in range(edges):
        factor = edge_weights[edge] / edge_degree[edge]
        for vertex in range(vertices):
            if vertex_degree[vertex] == 0.0:
                continue
            middle[edge][vertex] = factor * incidence[vertex][edge] / (vertex_degree[vertex] ** 0.5)
    mixed = [[0.0 for _ in range(vertices)] for _ in range(vertices)]
    for vertex in range(vertices):
        scale = 0.0 if vertex_degree[vertex] == 0.0 else 1.0 / (vertex_degree[vertex] ** 0.5)
        for other in range(vertices):
            total = 0.0
            for edge in range(edges):
                total += incidence[vertex][edge] * middle[edge][other]
            mixed[vertex][other] = scale * total
    lap = [[0.0 for _ in range(vertices)] for _ in range(vertices)]
    for vertex in range(vertices):
        for other in range(vertices):
            lap[vertex][other] = (1.0 if vertex == other else 0.0) - mixed[vertex][other]
    return lap


def quadratic_form(lap, features):
    total = 0.0
    count = len(features)
    for col in range(len(features[0])):
        signal = [row[col] for row in features]
        mixed = [sum(lap[i][j] * signal[j] for j in range(count)) for i in range(count)]
        total += sum(signal[i] * mixed[i] for i in range(count))
    return total
