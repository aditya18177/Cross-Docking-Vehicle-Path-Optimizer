from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import random
import copy
import json

app = Flask(__name__)

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
# Algorithm Functions
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

def tabu_search(pickup_routes, delivery_routes, max_iter=100, tabu_tenure=10):
    tabu_list = []
    best_pickup = copy.deepcopy(pickup_routes)
    best_delivery = copy.deepcopy(delivery_routes)
    best_cost = evaluate_solution(best_pickup, best_delivery)
    current_delivery = copy.deepcopy(best_delivery)
    for it in range(max_iter):
        neighborhood = []
        for i in range(num_vehicles):
            for j in range(num_vehicles):
                if i == j: continue
                for a in range(len(current_delivery[i])):
                    for b in range(len(current_delivery[j])):
                        move = ((i, a), (j, b))
                        if move in tabu_list: continue
                        neighbor = swap_nodes(current_delivery, i, j, a, b)
                        cost = evaluate_solution(best_pickup, neighbor)
                        neighborhood.append((cost, neighbor, move))
        if not neighborhood: break
        neighborhood.sort(key=lambda x: x[0])
        best_neighbor_cost, best_neighbor_delivery, best_move = neighborhood[0]
        if best_neighbor_cost < best_cost:
            best_cost = best_neighbor_cost
            best_delivery = best_neighbor_delivery
        current_delivery = best_neighbor_delivery
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
    return best_pickup, best_delivery, best_cost

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    
    # Run optimization algorithm
    pickup_routes, delivery_routes = create_initial_solution()
    opt_pickup, opt_delivery, total_cost = tabu_search(pickup_routes, delivery_routes)
    
    # Prepare response data
    response = {
        'pickup_routes': opt_pickup,
        'delivery_routes': opt_delivery,
        'total_cost': total_cost,
        'nodes': nodes_df.to_dict(orient='records'),
        'distance_matrix': distance_df.to_dict()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)