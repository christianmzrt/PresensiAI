import time
import heapq
from collections import deque
import numpy as np

def euclidean_dist(emb1, emb2):
    return float(np.linalg.norm(emb1 - emb2))

class FaceGraph:
    """
    Membangun graf jaringan kemiripan wajah (Face Similarity Graph)
    berdasarkan basis data mahasiswa untuk dieksplorasi algoritma pencarian.
    """
    def __init__(self, db_records):
        self.nodes = {}       # id -> record
        self.adj = {}         # id -> list of neighbor ids
        self.embeddings = {}  # id -> numpy embedding
        self._build_graph(db_records)

    def _build_graph(self, db_records):
        if not db_records:
            return

        for idx, rec in enumerate(db_records):
            node_id = f"node_{idx}_{rec['nim']}"
            self.nodes[node_id] = {
                "id": node_id,
                "nim": rec["nim"],
                "nama": rec["nama"],
                "mahasiswa_id": rec["mahasiswa_id"]
            }
            self.embeddings[node_id] = rec["embedding"]
            self.adj[node_id] = []

        node_keys = list(self.nodes.keys())
        n = len(node_keys)
        
        # Hubungkan simpul-simpul dalam graf berdasarkan k-Nearest Neighbors
        # agar membentuk struktur topologi graf yang valid untuk ditelusuri
        for i in range(n):
            u = node_keys[i]
            distances = []
            for j in range(n):
                if i != j:
                    v = node_keys[j]
                    d = euclidean_dist(self.embeddings[u], self.embeddings[v])
                    distances.append((d, v))
            
            # Hubungkan ke 2 tetangga terdekat (k=2) atau buat jalur rantai jika simpul sedikit
            distances.sort(key=lambda x: x[0])
            for d, v in distances[:max(2, min(3, n - 1))]:
                if v not in self.adj[u]:
                    self.adj[u].append(v)
                if u not in self.adj[v]:
                    self.adj[v].append(u)

            # Pastikan graf terhubung (connected graph)
            if i < n - 1:
                next_node = node_keys[i + 1]
                if next_node not in self.adj[u]:
                    self.adj[u].append(next_node)
                if u not in self.adj[next_node]:
                    self.adj[next_node].append(u)


def search_bfs(graph, start_node, query_embedding, threshold=0.90):
    """
    Breadth-First Search (BFS)
    Strategi: Uninformed Search menggunakan Queue (FIFO).
    Menelusuri simpul secara melebar lapis demi lapis.
    """
    t0 = time.perf_counter()
    if not graph.nodes or start_node not in graph.nodes:
        return {"found": False, "visited": [], "path": [], "time_ms": 0, "target": None}

    queue = deque([[start_node]])
    visited = set([start_node])
    visit_order = []

    while queue:
        path = queue.popleft()
        curr = path[-1]
        visit_order.append(curr)

        # Evaluasi apakah simpul saat ini cocok dengan query wajah
        dist = euclidean_dist(graph.embeddings[curr], query_embedding)
        if dist <= threshold:
            t1 = time.perf_counter()
            return {
                "algorithm": "Breadth-First Search (BFS)",
                "found": True,
                "target": graph.nodes[curr],
                "distance": round(dist, 4),
                "visited": visit_order,
                "path": path,
                "nodes_expanded": len(visit_order),
                "time_ms": round((t1 - t0) * 1000, 3)
            }

        for neighbor in graph.adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    t1 = time.perf_counter()
    return {
        "algorithm": "Breadth-First Search (BFS)",
        "found": False,
        "target": None,
        "distance": None,
        "visited": visit_order,
        "path": [],
        "nodes_expanded": len(visit_order),
        "time_ms": round((t1 - t0) * 1000, 3)
    }


def search_dfs(graph, start_node, query_embedding, threshold=0.90):
    """
    Depth-First Search (DFS)
    Strategi: Uninformed Search menggunakan Stack (LIFO).
    Menelusuri satu cabang sampai ke kedalaman maksimal sebelum backtracking.
    """
    t0 = time.perf_counter()
    if not graph.nodes or start_node not in graph.nodes:
        return {"found": False, "visited": [], "path": [], "time_ms": 0, "target": None}

    stack = [[start_node]]
    visited = set()
    visit_order = []

    while stack:
        path = stack.pop()
        curr = path[-1]

        if curr not in visited:
            visited.add(curr)
            visit_order.append(curr)

            # Evaluasi simpul
            dist = euclidean_dist(graph.embeddings[curr], query_embedding)
            if dist <= threshold:
                t1 = time.perf_counter()
                return {
                    "algorithm": "Depth-First Search (DFS)",
                    "found": True,
                    "target": graph.nodes[curr],
                    "distance": round(dist, 4),
                    "visited": visit_order,
                    "path": path,
                    "nodes_expanded": len(visit_order),
                    "time_ms": round((t1 - t0) * 1000, 3)
                }

            # Masukkan tetangga ke stack
            for neighbor in reversed(graph.adj.get(curr, [])):
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)

    t1 = time.perf_counter()
    return {
        "algorithm": "Depth-First Search (DFS)",
        "found": False,
        "target": None,
        "distance": None,
        "visited": visit_order,
        "path": [],
        "nodes_expanded": len(visit_order),
        "time_ms": round((t1 - t0) * 1000, 3)
    }


def search_greedy(graph, start_node, query_embedding, threshold=0.90):
    """
    Greedy Best-First Search
    Strategi: Informed Search menggunakan Heuristik h(n).
    Fungsi Evaluasi: f(n) = h(n), di mana h(n) = Euclidean Distance ke query embedding.
    Selalu memilih simpul dengan estimasi jarak terdekat ke target.
    """
    t0 = time.perf_counter()
    if not graph.nodes or start_node not in graph.nodes:
        return {"found": False, "visited": [], "path": [], "time_ms": 0, "target": None}

    # Priority Queue: (h(n), counter, path)
    counter = 0
    start_h = euclidean_dist(graph.embeddings[start_node], query_embedding)
    pq = [(start_h, counter, [start_node])]
    visited = set()
    visit_order = []

    while pq:
        h_val, _, path = heapq.heappop(pq)
        curr = path[-1]

        if curr in visited:
            continue
        visited.add(curr)
        visit_order.append(curr)

        dist = euclidean_dist(graph.embeddings[curr], query_embedding)
        if dist <= threshold:
            t1 = time.perf_counter()
            return {
                "algorithm": "Greedy Best-First Search",
                "found": True,
                "target": graph.nodes[curr],
                "distance": round(dist, 4),
                "visited": visit_order,
                "path": path,
                "nodes_expanded": len(visit_order),
                "time_ms": round((t1 - t0) * 1000, 3)
            }

        for neighbor in graph.adj.get(curr, []):
            if neighbor not in visited:
                neighbor_h = euclidean_dist(graph.embeddings[neighbor], query_embedding)
                counter += 1
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (neighbor_h, counter, new_path))

    t1 = time.perf_counter()
    return {
        "algorithm": "Greedy Best-First Search",
        "found": False,
        "target": None,
        "distance": None,
        "visited": visit_order,
        "path": [],
        "nodes_expanded": len(visit_order),
        "time_ms": round((t1 - t0) * 1000, 3)
    }


def search_astar(graph, start_node, query_embedding, threshold=0.90):
    """
    A* Search (A-Star)
    Strategi: Informed Search optimal menggunakan f(n) = g(n) + h(n).
    g(n) = biaya aktual kedalaman langkah pada graf (step cost).
    h(n) = nilai heuristik Euclidean Distance ke query embedding.
    """
    t0 = time.perf_counter()
    if not graph.nodes or start_node not in graph.nodes:
        return {"found": False, "visited": [], "path": [], "time_ms": 0, "target": None}

    counter = 0
    g_score = {start_node: 0.0}
    start_h = euclidean_dist(graph.embeddings[start_node], query_embedding)
    start_f = g_score[start_node] + start_h

    # Priority Queue: (f(n), counter, path)
    pq = [(start_f, counter, [start_node])]
    visited = set()
    visit_order = []

    while pq:
        f_val, _, path = heapq.heappop(pq)
        curr = path[-1]

        if curr in visited:
            continue
        visited.add(curr)
        visit_order.append(curr)

        dist = euclidean_dist(graph.embeddings[curr], query_embedding)
        if dist <= threshold:
            t1 = time.perf_counter()
            return {
                "algorithm": "A* Search (A-Star)",
                "found": True,
                "target": graph.nodes[curr],
                "distance": round(dist, 4),
                "visited": visit_order,
                "path": path,
                "nodes_expanded": len(visit_order),
                "time_ms": round((t1 - t0) * 1000, 3)
            }

        curr_g = g_score[curr]
        for neighbor in graph.adj.get(curr, []):
            tentative_g = curr_g + 1.0  # Biaya perpindahan antar simpul (step cost = 1.0)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                h_val = euclidean_dist(graph.embeddings[neighbor], query_embedding)
                f_score = tentative_g + h_val
                counter += 1
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (f_score, counter, new_path))

    t1 = time.perf_counter()
    return {
        "algorithm": "A* Search (A-Star)",
        "found": False,
        "target": None,
        "distance": None,
        "visited": visit_order,
        "path": [],
        "nodes_expanded": len(visit_order),
        "time_ms": round((t1 - t0) * 1000, 3)
    }


def compare_all_search(db_records, query_embedding, threshold=0.90):
    """
    Menjalankan ke-4 algoritma (BFS, DFS, Greedy, A*) pada graf database wajah
    dan membandingkan performa metriknya (Langkah ekspansi, waktu, dan jalur).
    """
    if not db_records:
        return {"error": "Database mahasiswa masih kosong."}

    graph = FaceGraph(db_records)
    start_node = list(graph.nodes.keys())[0]

    res_bfs = search_bfs(graph, start_node, query_embedding, threshold)
    res_dfs = search_dfs(graph, start_node, query_embedding, threshold)
    res_greedy = search_greedy(graph, start_node, query_embedding, threshold)
    res_astar = search_astar(graph, start_node, query_embedding, threshold)

    # Format graph topology for visualization
    topology = {
        "nodes": [{"id": k, "label": f"{v['nama']} ({v['nim']})"} for k, v in graph.nodes.items()],
        "edges": []
    }
    added_edges = set()
    for u, neighbors in graph.adj.items():
        for v in neighbors:
            edge_id = tuple(sorted([u, v]))
            if edge_id not in added_edges:
                added_edges.add(edge_id)
                topology["edges"].append({"from": u, "to": v})

    return {
        "start_node": start_node,
        "topology": topology,
        "results": [res_bfs, res_dfs, res_greedy, res_astar]
    }
