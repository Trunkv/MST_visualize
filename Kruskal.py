import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- GRAPH OPTIONS ----------------
graph_options = {
    "Graph 1": {
        "edges": [
            ("A","B",4),("A","C",1),("A","D",5),
            ("B","C",2),("B","E",3),
            ("C","D",6),("C","F",7),
            ("D","F",4),("E","F",5),("E","G",2),
            ("F","G",3),("F","H",6),("G","H",1)
        ],
        "pos":{
            "A":(-2,2),"B":(-3,1),"C":(-1,1),"D":(0,2),
            "E":(-2,0),"F":(0,0),"G":(-1,-1),"H":(1,1)
        }
    },
    "Graph 2": {
        "edges":[
            ("A","B",5),("A","C",3),("A","D",6),
            ("B","C",2),("B","E",4),
            ("C","D",7),("C","F",3),
            ("D","F",2),("E","F",6),("E","G",3),
            ("F","G",4),("F","H",5),("G","H",2)
        ],
        "pos":{
            "A":(-2,2),"B":(-3,1),"C":(-1,1),"D":(0,2),
            "E":(-2,0),"F":(0,0),"G":(-1,-1),"H":(1,1)
        }
    },
    "Graph 3": {
        "edges":[
            ("A","B",6),("A","C",2),("A","D",4),
            ("B","C",3),("B","E",5),
            ("C","D",1),("C","F",4),
            ("D","F",2),("E","F",6),("E","G",3),
            ("F","G",2),("F","H",5),("G","H",4)
        ],
        "pos":{
            "A":(-2,2),"B":(-3,1),"C":(-1,1),"D":(0,2),
            "E":(-2,0),"F":(0,0),"G":(-1,-1),"H":(1,1)
        }
    }
}

# ---------------- UNION-FIND ----------------
class UnionFind:
    def __init__(self,nodes):
        self.parent = {node:node for node in nodes}
    def find(self,x):
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x
    def union(self,x,y):
        px,py = self.find(x),self.find(y)
        if px==py:
            return False
        self.parent[py]=px
        return True

# ---------------- KRUSKAL NORMAL ----------------
def kruskal_steps_normal(G):
    edges = sorted([(d['weight'],u,v) for u,v,d in G.edges(data=True)])
    mst=[]
    steps=[]
    for w,u,v in edges:
        # kiểm tra chu trình bằng DFS
        G_temp = nx.Graph()
        G_temp.add_edges_from([(a,b) for a,b,_ in mst])
        G_temp.add_edge(u,v)
        if nx.is_tree(G_temp) or not nx.has_path(G_temp,u,v):
            mst.append((u,v,w))
            steps.append((mst.copy(), (w,u,v), "add"))
        else:
            steps.append((mst.copy(), (w,u,v), "discard"))
    return steps

# ---------------- KRUSKAL UNION-FIND ----------------
def kruskal_steps_uf(G):
    edges = sorted([(d['weight'],u,v) for u,v,d in G.edges(data=True)])
    mst=[]
    steps=[]
    uf=UnionFind(G.nodes)
    for w,u,v in edges:
        if uf.union(u,v):
            mst.append((u,v,w))
            steps.append((mst.copy(), (w,u,v), "add"))
        else:
            steps.append((mst.copy(), (w,u,v), "discard"))
    return steps

# ---------------- DRAW FUNCTION ----------------
def draw_graph(G,pos,step,final_color="purple",figsize=(5,5)):
    plt.figure(figsize=figsize)
    mst, current, action = step
    edge_colors=[]
    for u,v,w in G.edges(data='weight'):
        if (u,v,w) in mst or (v,u,w) in mst:
            edge_colors.append(final_color if len(mst)==len(G.nodes)-1 else "green")
        elif (u,v,w)==current or (v,u,w)==current:
            edge_colors.append("green" if action=="add" else "red")
        else:
            edge_colors.append("gray")
    node_colors=["lightblue" if n in sum([[e[0],e[1]] for e in mst],[]) else "white" for n in G.nodes()]
    nx.draw(G,pos,with_labels=True,node_color=node_colors,edge_color=edge_colors,
            width=2,font_weight='bold',node_size=500,font_size=8)
    edge_labels=nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=8)
    st.pyplot(plt)

# ---------------- STREAMLIT ----------------
st.title("Kruskal's Algorithm Visualization")

mode = st.selectbox("Chọn chế độ:",["Kruskal Normal","Kruskal Union-Find","Compare"])
graph_name = st.selectbox("Chọn đồ thị:", list(graph_options.keys()))

# reset step nếu đổi mode hoặc đồ thị
if "last_mode" not in st.session_state or st.session_state.last_mode!=mode or \
   "last_graph" not in st.session_state or st.session_state.last_graph!=graph_name:
    st.session_state.step=0
    st.session_state.last_mode=mode
    st.session_state.last_graph=graph_name

if st.button("Next Step"):
    st.session_state.step +=1

def show_edge(current):
    st.write(tuple(current))

# ---------------- MAIN ----------------
data = graph_options[graph_name]
G = nx.Graph()
G.add_weighted_edges_from(data["edges"])

if mode=="Kruskal Normal":
    steps = kruskal_steps_normal(G)
    if st.session_state.step>=len(steps):
        st.session_state.step=len(steps)-1
    draw_graph(G,data["pos"],steps[st.session_state.step])
    st.write("Current edge:")
    show_edge(steps[st.session_state.step][1])

elif mode=="Kruskal Union-Find":
    steps = kruskal_steps_uf(G)
    if st.session_state.step>=len(steps):
        st.session_state.step=len(steps)-1
    draw_graph(G,data["pos"],steps[st.session_state.step])
    st.write("Current edge:")
    show_edge(steps[st.session_state.step][1])

else: # Compare
    steps1 = kruskal_steps_normal(G)
    steps2 = kruskal_steps_uf(G)
    max_step = max(len(steps1),len(steps2))
    if st.session_state.step>=max_step:
        st.session_state.step=max_step-1
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Kruskal Normal")
        draw_graph(G,data["pos"],steps1[st.session_state.step] if st.session_state.step<len(steps1) else steps1[-1])
        st.write("Current edge:")
        show_edge(steps1[st.session_state.step][1] if st.session_state.step<len(steps1) else steps1[-1][1])
    with col2:
        st.subheader("Kruskal Union-Find")
        draw_graph(G,data["pos"],steps2[st.session_state.step] if st.session_state.step<len(steps2) else steps2[-1])
        st.write("Current edge:")
        show_edge(steps2[st.session_state.step][1] if st.session_state.step<len(steps2) else steps2[-1][1])
