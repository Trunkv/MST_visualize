import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math

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

# ---------------- DRAW FUNCTION ----------------
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
    else:
        visited,mst,heap = step
        last_added = None
        if mst:
            last_added = mst[-1][1]  # lấy đỉnh vừa thêm
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

# ---------------- STREAMLIT ----------------
st.title("Prim's Algorithm Visualization")

mode=st.selectbox("Chọn chế độ:",["Lazy Prim","Eager Prim","Compare"])

# Reset step khi đổi mode
if "last_mode" not in st.session_state or st.session_state.last_mode!=mode:
    st.session_state.step=0
    st.session_state.last_mode=mode

if st.button("Next Step"):
    st.session_state.step += 1

# ---------------- MAIN ----------------
def show_heap(heap):
    for h in heap:
        st.write(tuple(h))  # hiển thị mỗi item 1 dòng, ngắn gọn

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
    show_heap(steps[st.session_state.step][2])

else:  # Compare
    data = graph_options["Compare"]
    G1 = nx.Graph()
    G1.add_weighted_edges_from(data["edges"])
    G2 = nx.Graph()
    G2.add_weighted_edges_from(data["edges"])

    steps1 = lazy_prim_steps(G1,"A")
    steps2 = eager_prim_steps(G2,"A")

    # Chỉ giới hạn từng thuật toán riêng
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
        show_heap(steps2[step_eager][2])
