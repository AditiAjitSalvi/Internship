import networkx as nx

G = nx.DiGraph()

edges = [
    (1, 2, 15.73),
    (2, 3, 7.88),
    (3, 3, 0.05),
    (3, 4, 7.88),
    (4, 5, 7.88),
    (5, 6, 7.88),
    (6, 7, 7.88),
    (7, 8, 7.88),
    (8, 9, 7.88),
    (9, 10, 7.88),
    (10, 11, 7.88)
]

for u, v, w in edges:
    G.add_edge(u, v, weight=w)

print(G.edges(data=True))
