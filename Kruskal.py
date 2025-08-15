import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math
import random
import time

# ---------------- HEXAGON / CUSTOM LAYOUT ----------------
def hex_layout(ring_nodes, center_nodes):
    pos = {}
    n = len(ring_nodes)
    for i, node in enumerate(ring_nodes):
        angle = 2 * math.pi * i / n
        pos[node] = (1.5 * math.cos(angle), 1.5 * math.sin(angle))
    for i, node in enumerate(center_nodes):
        pos[node] = (0, 0.5 * i)
    return pos

# ---------------- GRAPH OPTIONS ----------------
graph_options = {
    "Lazy Prim": {
        "edges": [
            ("A", "B", 10), ("A", "C", 1), ("A", "D", 4),
            ("B", "C", 3), ("B", "E", 0), ("C", "F", 8),
            ("C", "D", 2), ("D", "F", 2), ("D", "G", 7),
            ("E", "F", 1), ("E", "H", 8), ("F", "G", 6),
            ("F", "H", 9), ("G", "H", 12)
        ],
        "pos": hex_layout(["A","B","E","H","G","D"], ["F","C"])
    },
    "Eager Prim": {
        "edges": [
            ("A", "B", 9), ("A", "C", 0), ("A", "D", 5),
            ("B", "D", -2), ("B", "E", 3), ("B", "G", 4),
            ("C", "F", 6),
            ("D", "F", 2), ("D", "G", 3),
            ("E", "G", 6),
            ("F", "G", 1)
        ],
        "pos": hex_layout(["E","G","F","C","A","B"], ["D"])
    },
    "Kruskal": {
        "edges": [
            ("A","B",4), ("A","H",8), ("B","H",11),
            ("B","C",8), ("H","I",7), ("H","G",1),
            ("I","G",6), ("C","D",7), ("C","F",4),
            ("C","I",2), ("D","F",14), ("D","E",9),
            ("F","E",10), ("F","G",2), ("G","H",1)
        ],
        "pos": {
            "A": (0,3), "B": (1,4), "C": (3,4),
            "D": (4,3), "E": (4,1), "F": (3,1),
            "G": (1,1), "H": (0,1), "I": (2,3)
        }
    },
    "Compare": {
        "edges": [
            ("A", "B", 10), ("A", "C", 1), ("A", "D", 4),
            ("B", "C", 3), ("B", "E", 0), ("C", "F", 8),
            ("C", "D", 2), ("D", "F", 2), ("D", "G", 7),
            ("E", "F", 1), ("E", "H", 8), ("F", "G", 6),
            ("F", "H", 9), ("G", "H", 12)
        ],
        "pos": {
            "A": (0, 3), "B": (-1, 2), "C": (1, 2), "D": (2, 3),
            "E": (-1, 1), "F": (1, 1), "G": (2, 1), "H": (0, 0)
        }
    }
}

# ---------------- LAZY PRIM ----------------
def lazy_prim_steps(G, start):
    visited = set([start])
    steps = [(set(visited), [], None, None, [])]  # bước 0: chỉ có start node
    heap = [(d['weight'], start, v) for v,d in G[start].items()]
    heapq.heapify(heap)
    mst=[]
    while heap:
        w,u,v = heapq.heappop(heap)
        if v in visited:
            steps.append((set(visited), mst.copy(), (u,v,w), "discard", heap.copy()))
            continue
        mst.append((u,v,w))
        visited.add(v)
        steps.append((set(visited), mst.copy(), (u,v,w), "add", heap.copy()))
        for to,d in G[v].items():
            if to not in visited:
                heapq.heappush(heap,(d['weight'],v,to))
    return steps

# ---------------- EAGER PRIM ----------------
def eager_prim_steps(G, start):
    visited=set()
    dist={node:float('inf') for node in G.nodes}
    edge_to={}
    dist[start]=0
    pq=[(0,start)]
    mst=[]
    steps=[(set(),[],pq.copy())]  # bước 0: chỉ có start node
    while pq:
        _,u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u in edge_to:
            mst.append(edge_to[u])
        for v,d in G[u].items():
            w=d['weight']
            if v not in visited and w<dist[v]:
                dist[v]=w
                edge_to[v]=(u,v,w)
                heapq.heappush(pq,(w,v))
        steps.append((set(visited), mst.copy(), pq.copy()))
    return steps

# ---------------- KRUSKAL ----------------
def kruskal_steps(G):
    edges = sorted(G.edges(data='weight'), key=lambda x: x[2])
    parent = {node: node for node in G.nodes}
    rank = {node:0 for node in G.nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # path compression
            u = parent[u]
        return u

    def union(u,v):
        u_root, v_root = find(u), find(v)
        if u_root == v_root:
            return False
        if rank[u_root] < rank[v_root]:
            parent[u_root] = v_root
        else:
            parent[v_root] = u_root
            if rank[u_root] == rank[v_root]:
                rank[u_root] += 1
        return True

    mst = []
    steps = []
    for u,v,w in edges:
        clusters_before = {n: find(n) for n in G.nodes}
        if union(u,v):
            action = "add"
            mst.append((u,v,w))
        else:
            action = "discard"
        clusters_after = {n: find(n) for n in G.nodes}
        steps.append((mst.copy(), (u,v,w), action, clusters_before.copy(), clusters_after.copy(), edges))
    return steps

# ---------------- DRAW FUNCTIONS ----------------
def draw_graph(G,pos,step,algo,final_color="purple",figsize=(4,4)):
    plt.figure(figsize=figsize)
    if algo=="Lazy":
        visited,mst,current,action,heap = step
        edge_colors=[]
        for u,v,w in G.edges(data='weight'):
            if (u,v,w) in mst or (v,u,w) in mst:
                edge_colors.append(final_color if len(visited)==len(G) else "green")
            elif current and ((u,v,w)==current or (v,u,w)==current):
                edge_colors.append("green" if action=="add" else "red")
            else:
                edge_colors.append("gray")
        node_colors=[]
        for n in G.nodes():
            if current and n==current[1]:
                node_colors.append("orange")
            elif n in visited:
                node_colors.append("lightblue")
            else:
                node_colors.append("white")
    elif algo=="Eager":
        visited,mst,heap = step
        last_added = mst[-1][1] if mst else None
        node_colors=["orange" if n==last_added else "lightblue" if n in visited else "white"
                     for n in G.nodes()]
        edge_colors=[]
        for u,v,w in G.edges(data='weight'):
            if (u,v,w) in mst or (v,u,w) in mst:
                edge_colors.append(final_color if len(visited)==len(G) else "green")
            else:
                edge_colors.append("gray")
    nx.draw(G,pos,with_labels=True,node_color=node_colors,edge_color=edge_colors,width=2,font_weight='bold',node_size=500,font_size=8)
    edge_labels=nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=8)
    st.pyplot(plt)

def draw_kruskal(G, pos, step, figsize=(4,4)):
    plt.figure(figsize=figsize)
    mst, current, action, clusters_before, clusters_after, all_edges = step

    # Xác định xem MST đã hoàn thành chưa
    mst_done = len(set([n for e in mst for n in e[:2]])) == len(G.nodes)

    # Mapping node -> màu cụm MST
    cluster_colors = {}
    clusters_nodes = {}
    cmap = plt.cm.tab20.colors
    next_color_idx = 0

    # Gán màu cho MST cũ
    for u,v,w in mst:
        u_cluster = cluster_colors.get(u)
        v_cluster = cluster_colors.get(v)
        if u_cluster and v_cluster:
            if u_cluster != v_cluster:
                for n in clusters_nodes[v_cluster]:
                    cluster_colors[n] = u_cluster
                clusters_nodes[u_cluster].update(clusters_nodes[v_cluster])
                del clusters_nodes[v_cluster]
            color = u_cluster
        elif u_cluster:
            color = u_cluster
            cluster_colors[v] = color
            clusters_nodes[color].add(v)
        elif v_cluster:
            color = v_cluster
            cluster_colors[u] = color
            clusters_nodes[color].add(u)
        else:
            color = cmap[next_color_idx % len(cmap)]
            next_color_idx += 1
            cluster_colors[u] = cluster_colors[v] = color
            clusters_nodes[color] = set([u,v])

    # Cạnh hiện tại
    u,v,w = current
    if action=="add":
        u_cluster = cluster_colors.get(u)
        v_cluster = cluster_colors.get(v)
        if u_cluster and not v_cluster:
            color = u_cluster
            cluster_colors[v] = color
            clusters_nodes[color].add(v)
        elif v_cluster and not u_cluster:
            color = v_cluster
            cluster_colors[u] = color
            clusters_nodes[color].add(u)
        elif not u_cluster and not v_cluster:
            color = cmap[next_color_idx % len(cmap)]
            next_color_idx += 1
            cluster_colors[u] = cluster_colors[v] = color
            clusters_nodes[color] = set([u,v])
        else:
            color = u_cluster
            for n in clusters_nodes[v_cluster]:
                cluster_colors[n] = color
            clusters_nodes[color].update(clusters_nodes[v_cluster])
            del clusters_nodes[v_cluster]
    else:
        color = "red"

    # Node màu
    node_colors = []
    for n in G.nodes():
        if n in cluster_colors:
            if n == current[1] and action=="add":
                node_colors.append("orange")
            else:
                node_colors.append("lightblue")
        else:
            node_colors.append("white")

    # Edge màu cho toàn bộ graph
    full_edge_colors = []
    for u,v,w in G.edges(data='weight'):
        if mst_done and ((u,v,w) in mst or (v,u,w) in mst):
            full_edge_colors.append("purple")  # MST hoàn thành
        elif (u,v,w) in mst or (v,u,w) in mst:
            full_edge_colors.append(cluster_colors[u])
        elif (u,v,w)==current or (v,u,w)==current:
            full_edge_colors.append(color if action=="add" else "red")
        else:
            full_edge_colors.append("gray")  # chưa trong MST

    nx.draw(G,pos,with_labels=True,node_color=node_colors,edge_color=full_edge_colors,
            width=2,font_weight='bold',node_size=500,font_size=8)
    edge_labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=8)
    st.pyplot(plt)

    # Hiển thị edges sorted dạng (weight,u,v)
    sorted_edges = sorted(all_edges, key=lambda x: x[2])
    st.write("Edges sorted:")
    for u,v,w in sorted_edges:
        st.write((w,u,v))

# ---------------- LARGE GRAPH ----------------
# --- Helper: tạo đồ thị lớn, Sparse hoặc Dense ---
def generate_large_graph(graph_type="Sparse", seed=7):
    random.seed(seed)

    if graph_type == "Sparse":
        n = 150
        m = 2 * n  # Sparse nhưng liên thông
        G = nx.gnm_random_graph(n, m, seed=seed)

        # Đảm bảo liên thông
        while not nx.is_connected(G):
            seed += 1
            G = nx.gnm_random_graph(n, m, seed=seed)

    elif graph_type == "Dense":
        n = 50
        m = min(200, n * (n - 1) // 2)  # Dense ~200 cạnh
        G = nx.gnm_random_graph(n, m, seed=seed)

        # Đảm bảo liên thông
        while not nx.is_connected(G):
            seed += 1
            G = nx.gnm_random_graph(n, m, seed=seed)

    elif graph_type == "Disconnected":
        n = 60
        num_clusters = 3  # số thành phần rời rạc
        cluster_size = n // num_clusters
        G = nx.Graph()
        G.add_nodes_from(range(n))

        p_in = 0.15  # xác suất kết nối trong cụm
        for c in range(num_clusters):
            nodes = range(c * cluster_size, (c + 1) * cluster_size)
            for i in nodes:
                for j in nodes:
                    if i < j and random.random() < p_in:
                        G.add_edge(i, j)
        # Không tạo cạnh giữa các cụm => chắc chắn không liên thông

    else:
        raise ValueError("graph_type must be 'Sparse', 'Dense', or 'Disconnected'")

    # Gán tên node N0, N1, ...
    mapping = {i: f"N{i}" for i in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # Gán trọng số ngẫu nhiên
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 20)

    # Layout chung
    pos = nx.spring_layout(G, seed=seed, k=0.3, iterations=50)

    return G, pos


def draw_large_graph(G, pos, mst_edges=set(), current_edge=None, final=False, figsize=(4,4)):
    plt.figure(figsize=figsize)
    edge_colors=[]
    for u,v,d in G.edges(data=True):
        if final and ((u,v) in mst_edges or (v,u) in mst_edges):
            edge_colors.append("purple")
        elif (u,v) in mst_edges or (v,u) in mst_edges:
            edge_colors.append("green")
        elif current_edge and ((u,v)==current_edge or (v,u)==current_edge):
            edge_colors.append("orange")
        else:
            edge_colors.append("gray")
    nx.draw(G,pos,node_color="lightblue",edge_color=edge_colors,node_size=20,with_labels=False)
    st.pyplot(plt)

def run_large_graph_animation(G, delay=0.05):
    # Eager Prim
    steps_eager = eager_prim_steps(G,"N0")
    mst_edges_eager = set()
    # Kruskal
    steps_kruskal = kruskal_steps(G)
    mst_edges_kruskal = set()

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Eager Prim")
        for s in steps_eager:
            visited,mst,_ = s
            mst_edges_eager = set((u,v) for u,v,w in mst)
            draw_large_graph(G,pos,mst_edges=mst_edges_eager)
            time.sleep(delay)
    with col2:
        st.subheader("Kruskal")
        for s in steps_kruskal:
            mst,_,_,_,_,_ = s
            mst_edges_kruskal = set((u,v) for u,v,w in mst)
            draw_large_graph(G,pos,mst_edges=mst_edges_kruskal)
            time.sleep(delay)

def show_heap_filtered(heap, in_tree):
    seen = set()
    filtered = []
    for w, v in heap:
        if v not in seen and v not in in_tree:
            seen.add(v)
            filtered.append((w, v))
    for h in filtered:
        st.write(h)
# ---------------- STREAMLIT ----------------
st.title("Prim's / Kruskal Visualization")

mode=st.selectbox("Chọn chế độ:", ["Lazy Prim", "Eager Prim", "Kruskal", "Compare", "Large Graph Compare"])

# Reset step khi đổi mode
if "last_mode" not in st.session_state or st.session_state.last_mode!=mode:
    st.session_state.step=0
    st.session_state.last_mode=mode

if st.button("Next Step"):
    st.session_state.step += 1

def show_heap(heap):
    for h in heap:
        st.write(tuple(h))

# ---------------- MAIN ----------------
if mode=="Lazy Prim":
    data=graph_options["Lazy Prim"]
    G=nx.Graph()
    G.add_weighted_edges_from(data["edges"])
    steps=lazy_prim_steps(G,"A")
    if st.session_state.step>=len(steps):
        st.session_state.step=len(steps)-1
    draw_graph(G,data["pos"],steps[st.session_state.step],"Lazy")
    st.write("Heap:")
    show_heap(steps[st.session_state.step][4])

elif mode=="Eager Prim":
    data=graph_options["Eager Prim"]
    G=nx.Graph()
    G.add_weighted_edges_from(data["edges"])
    steps=eager_prim_steps(G,"A")
    if st.session_state.step>=len(steps):
        st.session_state.step=len(steps)-1
    draw_graph(G,data["pos"],steps[st.session_state.step],"Eager")
    st.write("Heap:")
    show_heap_filtered(steps[st.session_state.step][2], steps[st.session_state.step][1])

elif mode=="Kruskal":
    data = graph_options["Kruskal"]
    G = nx.Graph()
    G.add_weighted_edges_from(data["edges"])
    steps = kruskal_steps(G)
    if st.session_state.step >= len(steps):
        st.session_state.step = len(steps)-1
    draw_kruskal(G,data["pos"],steps[st.session_state.step])

elif mode=="Compare":
    data = graph_options["Compare"]
    G1 = nx.Graph()
    G1.add_weighted_edges_from(data["edges"])
    G2 = nx.Graph()
    G2.add_weighted_edges_from(data["edges"])

    steps1 = lazy_prim_steps(G1,"A")
    steps2 = eager_prim_steps(G2,"A")

    step_lazy = st.session_state.step
    step_eager = st.session_state.step
    if step_lazy >= len(steps1):
        step_lazy = len(steps1) - 1
    if step_eager >= len(steps2):
        step_eager = len(steps2) - 1

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lazy Prim")
        draw_graph(G1, data["pos"], steps1[step_lazy], "Lazy")
        st.write("Heap:")
        show_heap(steps1[step_lazy][4])
    with col2:
        st.subheader("Eager Prim")
        draw_graph(G2, data["pos"], steps2[step_eager], "Eager")
        st.write("Heap:")
        # Eager: heap ở index 2, nhưng lọc cho gọn
        show_heap_filtered(steps2[step_eager][2], steps2[step_eager][1])
# --- Large Graph Compare với lựa chọn Sparse / Dense ---
elif mode == "Large Graph Compare":
    import time

    graph_type = st.selectbox("Chọn loại đồ thị:", ["Sparse", "Dense", "Disconnected"])

    # Khởi tạo đồ thị lớn nếu chưa có hoặc type khác
    if ("large_graph" not in st.session_state) or (st.session_state.large_type != graph_type):
        st.session_state.large_graph, st.session_state.large_pos = generate_large_graph(graph_type=graph_type)
        st.session_state.large_type = graph_type

    G = st.session_state.large_graph
    pos = st.session_state.large_pos
    st.write(f"Đồ thị {graph_type}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if st.button("Start Comparison"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Eager Prim")
            time_ph_eager = st.empty()
            fig_eager, ax_eager = plt.subplots(figsize=(6,6))
            ph_eager_plot = st.empty()
        with col2:
            st.subheader("Kruskal")
            time_ph_kruskal = st.empty()
            fig_kruskal, ax_kruskal = plt.subplots(figsize=(6,6))
            ph_kruskal_plot = st.empty()

        # Tính thời gian thực cho thuật toán
        t0 = time.time()
        steps_eager = eager_prim_steps(G, "N0")
        compute_time_eager = time.time() - t0

        t0 = time.time()
        steps_kruskal = kruskal_steps(G)
        compute_time_kruskal = time.time() - t0

        i_eager = 0; i_kruskal = 0
        done_eager = False; done_kruskal = False
        mst_edges_eager = set(); mst_edges_kruskal = set()
        KRUSKAL_FRAME_SKIP = max(1, len(steps_kruskal)//200)

        time_ph_eager.write("⏳ Eager Prim: đang chạy…")
        time_ph_kruskal.write("⏳ Kruskal: đang chạy…")

        while not (done_eager and done_kruskal):
            # Eager Prim frame
            if not done_eager:
                if i_eager < len(steps_eager):
                    visited, mst, _ = steps_eager[i_eager]
                    new_edge = mst[-1][:2] if mst else None
                    if new_edge:
                        mst_edges_eager.add(tuple(new_edge))
                    ax_eager.clear()
                    edge_colors = []
                    for u,v,d in G.edges(data=True):
                        if (u,v) in mst_edges_eager or (v,u) in mst_edges_eager:
                            if new_edge and (((u,v)==new_edge) or ((v,u)==new_edge)):
                                edge_colors.append("orange")
                            else:
                                edge_colors.append("green")
                        else:
                            edge_colors.append("gray")
                    nx.draw(G,pos,node_color="lightblue",edge_color=edge_colors,
                            node_size=18,with_labels=False,ax=ax_eager)
                    ph_eager_plot.pyplot(fig_eager)
                    i_eager +=1
                else:
                    ax_eager.clear()
                    edge_colors = ["#800080" if (u,v) in mst_edges_eager or (v,u) in mst_edges_eager else "gray"
                                   for u,v,d in G.edges(data=True)]
                    nx.draw(G,pos,node_color="lightblue",edge_color=edge_colors,
                            node_size=18,with_labels=False,ax=ax_eager)
                    ph_eager_plot.pyplot(fig_eager)
                    time_ph_eager.write(f"✅ Eager Prim time: {compute_time_eager:.4f} s")
                    done_eager = True

            # Kruskal frame
            if not done_kruskal:
                if i_kruskal < len(steps_kruskal):
                    mst, current, action, *_ = steps_kruskal[i_kruskal]
                    new_edge = current[:2] if action=="add" else None
                    mst_edges_kruskal = set((x,y) for x,y,_ in mst)
                    ax_kruskal.clear()
                    edge_colors = []
                    for u,v,d in G.edges(data=True):
                        if (u,v) in mst_edges_kruskal or (v,u) in mst_edges_kruskal:
                            if new_edge and (((u,v)==new_edge) or ((v,u)==new_edge)):
                                edge_colors.append("orange")
                            else:
                                edge_colors.append("green")
                        else:
                            edge_colors.append("gray")
                    nx.draw(G,pos,node_color="lightblue",edge_color=edge_colors,
                            node_size=18,with_labels=False,ax=ax_kruskal)
                    ph_kruskal_plot.pyplot(fig_kruskal)
                    i_kruskal += KRUSKAL_FRAME_SKIP
                else:
                    ax_kruskal.clear()
                    edge_colors = ["#800080" if (u,v) in mst_edges_kruskal or (v,u) in mst_edges_kruskal else "gray"
                                   for u,v,d in G.edges(data=True)]
                    nx.draw(G,pos,node_color="lightblue",edge_color=edge_colors,
                            node_size=18,with_labels=False,ax=ax_kruskal)
                    ph_kruskal_plot.pyplot(fig_kruskal)
                    time_ph_kruskal.write(f"✅ Kruskal time: {compute_time_kruskal:.4f} s")
                    done_kruskal = True

            time.sleep(0.02)








