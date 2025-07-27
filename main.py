import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------
# Load Data
# ---------------------------
nodes_df = pd.read_csv("synthetic_nodes.csv")
distance_df = pd.read_csv("distance_matrix.csv", index_col=0)
distance_df.index = distance_df.index.astype(str)
distance_df.columns = distance_df.columns.astype(str)


# Parameters
num_vehicles = 3
crossdock_id = 0

# Node separation
suppliers = nodes_df[nodes_df['type'] == 'supplier']
customers = nodes_df[nodes_df['type'] == 'customer']

# ---------------------------
# Utility Functions
# ---------------------------
def create_initial_solution():
    pickup_routes = [[] for _ in range(num_vehicles)]
    delivery_routes = [[] for _ in range(num_vehicles)]
    s_ids = suppliers['node_id'].tolist()
    c_ids = customers['node_id'].tolist()
    random.shuffle(s_ids)
    random.shuffle(c_ids)
    for i, sid in enumerate(s_ids):
        pickup_routes[i % num_vehicles].append(sid)
    for i, cid in enumerate(c_ids):
        delivery_routes[i % num_vehicles].append(cid)
    return pickup_routes, delivery_routes

def compute_route_cost(route):
    cost = 0
    last = crossdock_id
    for nid in route:
        cost += distance_df.loc[str(last), str(nid)]
        last = nid
    cost += distance_df.loc[str(last), str(crossdock_id)]
    return cost

def evaluate_solution(pickup_routes, delivery_routes):
    total = 0
    for pr, dr in zip(pickup_routes, delivery_routes):
        total += compute_route_cost(pr) + compute_route_cost(dr)
    return total

def swap_nodes(routes, i, j, a, b):
    new_routes = copy.deepcopy(routes)
    new_routes[i][a], new_routes[j][b] = new_routes[j][b], new_routes[i][a]
    return new_routes

# Example improvement to tabu_search with aspiration criteria
def tabu_search(pickup_routes, delivery_routes, max_iter=100, tabu_tenure=10):
    tabu_list = []
    best_pickup = copy.deepcopy(pickup_routes)
    best_delivery = copy.deepcopy(delivery_routes)
    best_cost = evaluate_solution(best_pickup, best_delivery)
    current_pickup = copy.deepcopy(best_pickup)
    current_delivery = copy.deepcopy(best_delivery)
    
    for it in range(max_iter):
        neighborhood = []
        # Evaluate both pickup and delivery neighborhoods
        for routes, route_type in [(current_delivery, 'delivery'), (current_pickup, 'pickup')]:
            for i in range(num_vehicles):
                for j in range(num_vehicles):
                    if i == j: continue
                    for a in range(len(routes[i])):
                        for b in range(len(routes[j])):
                            move = ((route_type, i, a), (route_type, j, b))
                            if move in tabu_list: continue
                            if route_type == 'pickup':
                                neighbor_pickup = swap_nodes(routes, i, j, a, b)
                                neighbor_delivery = current_delivery
                            else:
                                neighbor_pickup = current_pickup
                                neighbor_delivery = swap_nodes(routes, i, j, a, b)
                            cost = evaluate_solution(neighbor_pickup, neighbor_delivery)
                            neighborhood.append((cost, neighbor_pickup, neighbor_delivery, move))
        
        if not neighborhood: break
        neighborhood.sort(key=lambda x: x[0])
        
        # Aspiration criteria - accept if better than best found
        best_neighbor_cost, best_neighbor_pickup, best_neighbor_delivery, best_move = neighborhood[0]
        if best_neighbor_cost < best_cost:
            best_cost = best_neighbor_cost
            best_pickup = best_neighbor_pickup
            best_delivery = best_neighbor_delivery
        
        current_pickup, current_delivery = best_neighbor_pickup, best_neighbor_delivery
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
    return best_pickup, best_delivery, best_cost

# ---------------------------
# Run Optimization
# ---------------------------
pickup_routes, delivery_routes = create_initial_solution()
opt_pickup, opt_delivery, total_cost = tabu_search(pickup_routes, delivery_routes)

# ---------------------------
# Plot Directed Graph
# ---------------------------
def plot_routes(nodes, pickup_routes, delivery_routes, cost, distance_df):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    pos = {}
    labels = {}
    node_colors = []

    # Vehicle-specific colors and line styles
    vehicle_colors = ['red', 'blue', 'green']  # Colors for vehicles 0, 1, 2
    vehicle_styles = ['solid', 'dashed', 'dotted']  # Line styles for vehicles 0, 1, 2
    pickup_linewidth = 2
    delivery_linewidth = 3

    # Add nodes and color by type
    for _, row in nodes.iterrows():
        node_id = row['node_id']
        G.add_node(node_id)
        pos[node_id] = (row['x'], row['y'])
        labels[node_id] = f"{node_id}"

        if row['type'] == 'supplier':
            node_colors.append('orange')   # 🟧 Supplier
        elif row['type'] == 'customer':
            node_colors.append('lightgreen')    # 🟩 Customer
        else:
            node_colors.append('skyblue')  # 🔵 Cross-dock

    edge_labels = {}
    
    # We'll draw edges separately for each vehicle to apply different styles
    for v in range(len(pickup_routes)):
        pickup = pickup_routes[v]
        delivery = delivery_routes[v]

        # ---- Pickup phase: crossdock ➝ suppliers ➝ crossdock
        if pickup:
            path = [0] + pickup + [0]
            for i in range(len(path) - 1):
                u, v_ = path[i], path[i+1]
                dist = float(distance_df.loc[str(u), str(v_)])
                G.add_edge(u, v_)
                edge_labels[(u, v_)] = f"{dist:.1f}"
                
                # Draw pickup edge with vehicle-specific style
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v_)],
                    edge_color=vehicle_colors[v],
                    style=vehicle_styles[v],
                    width=pickup_linewidth,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=10
                )

        # ---- Delivery phase: crossdock ➝ customers ➝ crossdock
        if delivery:
            path = [0] + delivery + [0]
            for i in range(len(path) - 1):
                u, v_ = path[i], path[i+1]
                dist = float(distance_df.loc[str(u), str(v_)])
                G.add_edge(u, v_)
                edge_labels[(u, v_)] = f"{dist:.1f}"
                
                # Draw delivery edge with vehicle-specific style (thicker line)
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v_)],
                    edge_color=vehicle_colors[v],
                    style=vehicle_styles[v],
                    width=delivery_linewidth,
                    arrows=True,
                    arrowstyle='-|>',
                    arrowsize=15
                )

    # Draw nodes and labels
    nx.draw(G, pos, node_color=node_colors, with_labels=True, labels=labels,
            node_size=700, font_size=10)

    # Draw distances on edges
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_color='black', font_size=9)

    # Create legend for vehicles
    legend_elements = [
        plt.Line2D([0], [0], color=vehicle_colors[0], linestyle=vehicle_styles[0], 
                   lw=2, label='Vehicle 1'),
        plt.Line2D([0], [0], color=vehicle_colors[1], linestyle=vehicle_styles[1], 
                   lw=2, label='Vehicle 2'),
        plt.Line2D([0], [0], color=vehicle_colors[2], linestyle=vehicle_styles[2], 
                   lw=2, label='Vehicle 3'),
        plt.Line2D([0], [0], color='black', linestyle='solid', 
                   lw=pickup_linewidth, label='Pickup Route'),
        plt.Line2D([0], [0], color='black', linestyle='solid', 
                   lw=delivery_linewidth, label='Delivery Route')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(f"Optimized Routes with Total Cost: {cost:.2f}")
    plt.show()




plot_routes(nodes_df, opt_pickup, opt_delivery, total_cost, distance_df)

