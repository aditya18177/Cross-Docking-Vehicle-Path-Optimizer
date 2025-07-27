import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import networkx as nx
import itertools

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

def plot_routes(nodes, pickup_routes, delivery_routes, cost, distance_df, title_suffix=""):
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
    
    # Draw edges separately for each vehicle to apply different styles
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
    plt.title(f"Brute/Worst Possible Routes with Total Cost: {cost:.2f} {title_suffix}")
    plt.show()

# ---------------------------
# Brute Force Implementation
# ---------------------------
def brute_force_max_cost():
    # Get all supplier and customer IDs
    supplier_ids = suppliers['node_id'].tolist()
    customer_ids = customers['node_id'].tolist()
    
    # Generate all possible partitions for suppliers
    supplier_partitions = []
    for partition in itertools.product(range(num_vehicles), repeat=len(supplier_ids)):
        routes = [[] for _ in range(num_vehicles)]
        for node_idx, truck_idx in enumerate(partition):
            routes[truck_idx].append(supplier_ids[node_idx])
        supplier_partitions.append(routes)
    
    # Generate all possible partitions for customers
    customer_partitions = []
    for partition in itertools.product(range(num_vehicles), repeat=len(customer_ids)):
        routes = [[] for _ in range(num_vehicles)]
        for node_idx, truck_idx in enumerate(partition):
            routes[truck_idx].append(customer_ids[node_idx])
        customer_partitions.append(routes)
    
    max_cost = -1
    worst_pickup = None
    worst_delivery = None
    
    # We'll limit the brute force to first 1000 combinations for performance
    max_combinations = 1000
    evaluated = 0
    
    for pickup in supplier_partitions[:max_combinations]:
        for delivery in customer_partitions[:max_combinations]:
            cost = evaluate_solution(pickup, delivery)
            if cost > max_cost:
                max_cost = cost
                worst_pickup = pickup
                worst_delivery = delivery
            evaluated += 1
            if evaluated >= max_combinations:
                break
        if evaluated >= max_combinations:
            break
    
    return worst_pickup, worst_delivery, max_cost

# ---------------------------
# Run Optimization
# ---------------------------
print("Finding worst possible routes (brute force approach)...")
worst_pickup, worst_delivery, max_cost = brute_force_max_cost()

print("\nWorst Possible Solution Found:")
print(f"Total Cost: {max_cost:.2f}")
print("\nPickup Routes:")
for i, route in enumerate(worst_pickup):
    print(f"Truck {i+1}: {route}")
print("\nDelivery Routes:")
for i, route in enumerate(worst_delivery):
    print(f"Truck {i+1}: {route}")

# ---------------------------
# Plot Worst Routes
# ---------------------------
plot_routes(nodes_df, worst_pickup, worst_delivery, max_cost, distance_df, "(Worst Case)")