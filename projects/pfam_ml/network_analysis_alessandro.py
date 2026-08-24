PREDICTED_NET_FILE = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/Results/Prediction/Healthy_Unspecified/Fuzzy_ClusterClusterInteractions.txt"
NET_INFO_FILE = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/Results/Prediction/Healthy_Unspecified/Aggregated_results_Healthy_Unspecified.xlsx"
RANDOM_NETS_FILE = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/random_networks"
import os
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import json

from collections import defaultdict
from plotly.subplots import make_subplots

# Set plotly theme to white
pio.templates.default = "plotly_white"

PLOTS_DIR = "./figs"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def network_construction(resa, resb):
    # 2. Node list and attributes
    node_dict = {}
    lista = (list(resa["HumanCluster"]))
    listb = (list(resb["BacterialCluster"]))
    nodes_list = lista + listb

    # Set the fc_ID as index in df and only keep Degree column
    node_dicta = resa.set_index("HumanCluster")[["Degree"]]
    node_dicta["Species"] = "Human"
    node_dictb = resb.set_index("BacterialCluster")[["Degree"]] 
    node_dictb["Species"] = "Bacteria"

    # Make df to dict (if not orient=index, the dict is transposed)
    dicta = node_dicta.to_dict(orient="index")
    dictb = node_dictb.to_dict(orient="index")

    # Fill in a dict: key: fc_ID, value: degree
    # Όταν κάνεις το df --> dict, k γίνεται το index του df (primary key), αλλά κάθε row του df γίνεται
    # ένα μικρό dict (άρα σύνολο έχω nested dict) με key name(s) τα names(s) από τα columns και value(s) τα
    # αντίστοιχα
    for k,v in dicta.items():
        node_dict[k] = v["Degree"]
    for k,v in dictb.items():
        node_dict[k] = v["Degree"]

    # 3. Edge list
    with open(PREDICTED_NET_FILE) as file:
        edges_list = []
        for line in file:
            a = line.rstrip()
            a = a.split("\t")
            p1 = a[0]
            p2 = a[1]
            edge = [p1, p2]
            edges_list.append(edge)

    # 4. Network construction
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edges_list)
    nx.set_node_attributes(G, node_dict, name="degree")
    return G

def network_without_top_nodes_construction(G, percentage=0.01):
    degree = dict(G.degree())
    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [x[0] for x in sorted_degree[:int(len(sorted_degree)*percentage)]]

    # Copy the network
    G1 = G.copy()
    
    # Remove the top nodes
    G1.remove_nodes_from(top_nodes)

    return G1

def degree_distribution(resa, resb):
    # Set the fc_ID as index in df and only keep Degree column
    node_dicta = resa.set_index("HumanCluster")[["Degree"]]
    node_dicta["Species"] = "Human"
    node_dictb = resb.set_index("BacterialCluster")[["Degree"]] 
    node_dictb["Species"] = "Bacteria"

    # Merge 
    node_dict = pd.concat([node_dicta, node_dictb])
    fig = px.histogram(node_dict, x="Degree", color="Species", nbins=20)
    fig.update_layout(xaxis_title="Degree", yaxis_title="Number of nodes", font={"family": "Times New Roman", "size": 20})
    pio.write_image(fig, os.path.join(PLOTS_DIR, "degree_distribution.svg"), width=800, height=600, scale = 1)

def plot_degree_and_pairs(G,m):
    dict_edge = {}
    for edge in G.edges:
       dict_edge[edge] = {"Human Degree" : G.degree[edge[0]], "Bacterial Degree" : G.degree[edge[1]]}
    df = pd.DataFrame(dict_edge).T
    # Scatter plot: x axis Human degree, y axis Bacterial degree
    fig = px.scatter(df, x="Human Degree", y="Bacterial Degree")
    fig.update_layout(title="Degree of interactions", xaxis_title="Human Degree", yaxis_title="Bacterial Degree", font={"family": "Times New Roman", "size": 20})
    pio.write_image(fig, "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/degree_of_interactions_{}.png".format(m), width=800, height=600, scale=2)
    
def assortativity(G,m):
    assortativity = nx.degree_assortativity_coefficient(G)
    print("Assortativity of {}: {}".format(m, assortativity))

def knn(G):
    knn_values = nx.average_degree_connectivity(G)
    knn_df = pd.DataFrame(knn_values.items(), columns=["Degree", "Average degree connectivity"])
    fig = go.Scatter(
        x=knn_df["Degree"],
        y=knn_df["Average degree connectivity"],
        mode="markers",
        name="Knn Plot"
    )
    return fig

def knn_avg_degree_connectivity(G,G1):

    g_plot = knn(G)
    g1_plot = knn(G1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "a: Full Network", 
        "b: Network Excluding Top 1% Nodes"
    ))

    # Add the plots to subfigures
    fig.add_trace(g_plot, row=1, col=1)
    fig.add_trace(g1_plot, row=1, col=2)

    # Update layout
    fig.update_layout(
        font={"family": "Times New Roman", "size": 20},
        template="plotly_white",
        showlegend=False
    )

    # x and y axis title in both traces
    fig.update_xaxes(title_text="Degree", row=1, col=1)
    fig.update_xaxes(title_text="Degree", row=1, col=2)
    fig.update_yaxes(title_text="Average Degree Connectivity", row=1, col=1)
    fig.update_yaxes(title_text="Average Degree Connectivity", row=1, col=2)

    fig.update_annotations(font=dict(family="Times New Roman", size=20))

    # Save the figure
    pio.write_image(fig, os.path.join(PLOTS_DIR, "knn_avg_degree_connectivity.svg"), width=1200, height=800, scale = 1)

def correlation_profile(G,m):
    # Dictionary: key--> degree-degree pairs, value--> number of pairs
    dict_pairs = {}
    for edge in G.edges:
        pair = (G.degree[edge[0]], G.degree[edge[1]])
        dict_pairs[pair]= dict_pairs.get(pair, 0) + 1
    for k,v in dict_pairs.items():
        dict_pairs[k] = v/len(G.edges)
    
def randomize_network(G):
    # Create randomized network with the same degree distribution (and assortativity = 0)
    random_net_dir = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/random_networks"
    os.makedirs(random_net_dir, exist_ok=True)
    for i in range(1000):
        print(i,"/1000")
        g_ran = nx.configuration_model([y[1] for y in G.degree()])
        print("Number of nodes in the network: {}".format(len(g_ran.nodes)))
        print("Number of edges in the network: {}".format(len(g_ran.edges)))
        dict_pairs_ran = {}
        for edge in g_ran.edges:
            pair = (g_ran.degree[edge[0]], g_ran.degree[edge[1]])
            dict_pairs_ran[str(pair)]= dict_pairs_ran.get(pair, 0) + 1
        #for k,v in dict_pairs_ran.items():
        #    dict_pairs_ran[k] = v/len(g_ran.edges)
        random_net = {
            "nodes": len(g_ran.nodes),
            "edges": len(g_ran.edges),
            "frequency": dict_pairs_ran
        }
        
        with open(os.path.join(random_net_dir, "random_network_{}.json".format(i)), "w") as f:
            json.dump(random_net, f, indent=4)

def random_stats():
    # Change frequency to probability
    for index, file in enumerate (os.listdir(RANDOM_NETS_FILE)):
        try:
            print("File: {}/{}".format(index, len(os.listdir(RANDOM_NETS_FILE))))
            f = os.path.join(RANDOM_NETS_FILE, file)
            with open(f) as f1:
                random_net = json.load(f1)
                f1.close()
            if "probability" in random_net:
                del random_net["probability"]
            frequency = random_net["frequency"]
            for k,v in frequency.items():
                frequency[str(k)] = v/random_net["edges"]
            random_net["probability"] = frequency
            with open(f, "w") as f1:
                json.dump(random_net, f1, indent=4)
                f1.close()
        except Exception as e:
            print(e)

def aggregate_probs():
    for index, file in enumerate (os.listdir(RANDOM_NETS_FILE)):
        try:
            print("File: {}/{}".format(index, len(os.listdir(RANDOM_NETS_FILE))))
            f = os.path.join(RANDOM_NETS_FILE, file)
            with open(f) as f1:
                random_net = json.load(f1)
                f1.close()
            new_dict_prob = {}
            for k,v in random_net["probability"].items():
                # k as tuple of ints
                k = tuple(map(int, k.strip("()").split(",")))
                k = tuple(sorted(k))
                if k in new_dict_prob:
                    new_dict_prob[str(k)] += v
                    new_dict_prob[str(k)] = new_dict_prob[str(k)]/2
                else:
                    new_dict_prob[str(k)] = v

            random_net["probability"] = new_dict_prob
            with open(f, "w") as f1:
                json.dump(random_net, f1, indent=4)
                f1.close()

        except Exception as e:
            print(e)

def z_score(G):
    # Calculate the z-score for probability of degree-degree pairs in normal network and one randomized network
    # Z score = (Pk-Pk1) - (Pk_randomizek - Pk1_randomized)/std(in 1000 random networks)
    dict_pairs = {}
    prob_final = {}
    for edge in G.edges:
        pair = (G.degree[edge[0]], G.degree[edge[1]])
        # Sort the pair to avoid duplicates (ευθύ και ανάστροφο)
        pair = tuple(sorted(pair))
        dict_pairs[pair]= dict_pairs.get(pair, 0) + 1
    # For every pair in G, we will find all the posibilities in the 1000 random networks (for std calculation)
    for idx, (k,v) in enumerate(dict_pairs.items()):
        print("Degree-degree pair: {}/{}".format(idx,len(dict_pairs)))

        #prob_pairs[k] = v/len(G.edges)
        original_value =  v/len(G.edges)
        value_list = []
        for filename in os.listdir(RANDOM_NETS_FILE):
            f = os.path.join(RANDOM_NETS_FILE, filename)
            with open(f) as f:
                random_net = json.load(f)
                frequency = random_net["frequency"]
                # Calculate (a,b) and (b,a) frequencies --> avoid duplicates
                value_ab = 0
                value_ba = 0

                # Prepei oi pithanotites gia ta random na ypologizontai mia fora prin to loop kai oxi mesa sto loop gia ta pairs.
                # Aplws tha tis vroume oles tis pithanotites gia ta zeygaria sta 1000 random networks kai tha anaktame meta.

                if str(k) in frequency: 
                    value_ab = frequency[str(k)]
                if str(k[::-1]) in frequency:
                    value_ba = frequency[str(k[::-1])]
                # To every (a,b) pair frequency, add the (b,a) frequency
                value = (value_ab + value_ba)/random_net["edges"]
                value_list.append(value)
        # Calculate std of value_lisτ: std of every pair in the 1000 random networks
        std = np.std(value_list)
        # Calculate z-score
        random_prob = value
        probs_dict = {
            "Probability in G": original_value,
            "Probability in Random_Network_1000": random_prob,
            "Std": std,
            "Z-score": (original_value - random_prob)/std
        }
        print("Probability in G: {}".format(original_value))
        print("Probability in Random_Network_1000: {}".format(random_prob))
        print("Std: {}".format(std))
        print("Z-score: {}".format((original_value - random_prob)/std))
        print("-"*100)
        # Nested dictionary: key--> degree-degree pair, value--> dictionary with probabilities, std, z-score for every degree-degree pair
        prob_final[k] = probs_dict
    # Write to csv
    df = pd.DataFrame(prob_final).T
    df.to_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_score.csv")


def calculate_degree_pair_probabilities(G):
    degree_pair_counts = defaultdict(int)
    n_edges = G.number_of_edges()
    for u, v in G.edges():
        degree_u = G.degree[u]
        degree_v = G.degree[v]
        degree_pair_counts[(degree_u, degree_v)] += 1
        if degree_u != degree_v:
            degree_pair_counts[(degree_v, degree_u)] += 1
    degree_pair_probs = {k: v / (2 * n_edges) for k, v in degree_pair_counts.items()}
    return degree_pair_probs

def generate_random_networks_and_calculate_probabilities(G,net="init", n=1000):
    degree_sequence = [d for n, d in G.degree()]
    degree_pair_sums = defaultdict(float)
    degree_pair_sums_sq = defaultdict(float)
    original_probabilities = calculate_degree_pair_probabilities(G)

    for i in range(n):
        print("Random network {}/{}".format(i, n))
        random_net = nx.configuration_model(degree_sequence)
        random_net = nx.Graph(random_net)
        random_net.remove_edges_from(nx.selfloop_edges(random_net))
        random_probabilities = calculate_degree_pair_probabilities(random_net)

        for pair, prob in random_probabilities.items():
            degree_pair_sums[pair] += prob
            degree_pair_sums_sq[pair] += prob**2

    means = {k: v / n for k, v in degree_pair_sums.items()}
    stds = {k: np.sqrt(degree_pair_sums_sq[k] / n - means[k]**2) for k in means}

    # Calculate z-scores
    print("Calculating z-scoress")
    z_scores = {}
    for pair in original_probabilities:
        if pair in stds and stds[pair] > 0:
            z_scores[pair] = (original_probabilities[pair] - means.get(pair, 0)) / stds[pair]
        else:
            z_scores[pair] = 0  # If std is 0, z-score is not defined; we could choose to handle this differently

    if net == "init":
        # Save z-scores to a file. Convert pair to string to avoid issues with JSON keys
        with open("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_scores.json", "w") as f:
            json.dump({str(k): v for k, v in z_scores.items()}, f, indent=4)
    else:
        with open("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_scores_{}.json".format(net), "w") as f:
            json.dump({str(k): v for k, v in z_scores.items()}, f, indent=4)

def z_heatmap(net="init"):
    """
    Create a heatmap of z-scores for degree-degree pairs
    x: degree of the first node
    y: degree of the second node
    color: z-score
    """
    if net == "init":
        with open("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_scores.json") as f:
            z_scores = json.load(f)
    else:
        with open("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_scores_{}.json".format(net)) as f:
            z_scores = json.load(f)

    print("Min z-score: {}".format(min(z_scores.values())))
    print("Max z-score: {}".format(max(z_scores.values())))
    print("Number of z-scores: {}".format(len(z_scores)))

    degrees = set()
    for pair in z_scores.keys():
        # Pair to tuple
        pair = tuple(map(int, pair.strip("()").split(",")))
        degrees.add(pair[0])
        degrees.add(pair[1])

    degrees = sorted(list(degrees))
    
    # Create a matrix filled with NaNs (for degrees without a z-score)
    z_matrix = np.full((len(degrees), len(degrees)), np.nan)
    
    # Map from degree to its index in the matrix
    degree_to_idx = {degree: idx for idx, degree in enumerate(degrees)}
    
    # Fill the matrix with z-scores
    for pair, z in z_scores.items():
        # Pair to tuple
        pair = tuple(map(int, pair.strip("()").split(",")))
        d1, d2 = pair
        idx1 = degree_to_idx[d1]
        idx2 = degree_to_idx[d2]
        z_matrix[idx1, idx2] = z
        z_matrix[idx2, idx1] = z  # Symmetric for undirected graph

    # Create and show the heatmap. Z-score limits according to the min and max z-score
    fig = go.Figure(data=go.Heatmap(z=z_matrix, x=degrees, y=degrees, colorscale="Viridis",zmin=-1.34, zmax=2.38))
    fig.update_layout(xaxis_title="Degree of the second node", yaxis_title="Degree of the first node", font={"family": "Times New Roman", "size": 20})
    
    if net == "init":
        #pio.write_image(fig, "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_scores_heatmap.png", width=800, height=600, scale=2)
        pio.write_image(fig, os.path.join(PLOTS_DIR, "z_scores_heatmap.svg"), width=800, height=600, scale=1)
    else:
        #pio.write_image(fig, "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/z_scores_heatmap_{}.png".format(net), width=800, height=600, scale=2)
        pio.write_image(fig, os.path.join(PLOTS_DIR, "z_scores_heatmap_{}.svg".format(net)), width=800, height=600, scale=1)
    
def main():
      # 1. Read files: a. Human Clusters, b. Bacterial Clusters
    resa = pd.read_excel(NET_INFO_FILE, sheet_name = "FuzzyHumanClusters", index_col = 0, engine='openpyxl')
    resb = pd.read_excel(NET_INFO_FILE, sheet_name = "FuzzyBacterialClusters", index_col = 0, engine='openpyxl')
    
    # Network Construction
    G = network_construction(resa, resb)
  
    print("Number of nodes in the network: {}".format(len(G.nodes)))
    print("Number of nodes in the network: {}".format(len(G.edges)))

    G1 = network_without_top_nodes_construction(G, percentage=0.01)
    print("Number of nodes in the network: {}".format(len(G1.nodes)))
    print("Number of nodes in the network: {}".format(len(G1.edges)))

    # Network Analysis
    degree_distribution(resa, resb)
    assortativity(G,"G")
    assortativity(G1,"G1")

    knn_avg_degree_connectivity(G,G1)

    #plot_degree_and_pairs(G,"G")
    #plot_degree_and_pairs(G1,"G1")
    #randomize_network(G)
    #random_stats()
    #aggregate_probs()
    #z_score(G)

    #generate_random_networks_and_calculate_probabilities(G)
    z_heatmap()

    #generate_random_networks_and_calculate_probabilities(G1,"no_topk")
    #z_heatmap("no_topk")



if __name__ == "__main__":
    main()